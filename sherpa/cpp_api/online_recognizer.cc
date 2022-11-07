/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sherpa/cpp_api/online_recognizer.h"

#include <utility>

#include "nlohmann/json.hpp"
#include "sherpa/csrc/file_utils.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online-conv-emformer-transducer-model.h"
#include "sherpa/csrc/online-transducer-decoder.h"
#include "sherpa/csrc/online-transducer-greedy-search-decoder.h"
#include "sherpa/csrc/online-transducer-model.h"
#include "sherpa/csrc/symbol_table.h"

namespace sherpa {

std::string OnlineRecognitionResult::AsJsonString() const {
  using json = nlohmann::json;
  json j;
  j["text"] = text;

  // TODO(fangjun): The key in the json object should be kept
  // in sync with sherpa/bin/pruned_transducer_statelessX/streaming_server.py
  j["segment"] = segment;  // TODO(fangjun): Support endpointing
  j["final"] = is_final;
  return j.dump();
}

void OnlineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);

  po->Register("nn-model", &nn_model, "Path to the torchscript model");

  po->Register("tokens", &tokens, "Path to tokens.txt.");

  po->Register("use-gpu", &use_gpu,
               "true to use GPU for computation. false to use CPU.\n"
               "If true, it uses the first device. You can use the environment "
               "variable CUDA_VISIBLE_DEVICES to select which device to use.");

  po->Register("decoding-method", &decoding_method,
               "Decoding method to use. Possible values are: greedy_search, "
               "modified_beam_search");

  po->Register("num-active-paths", &num_active_paths,
               "Number of active paths for modified_beam_search. "
               "Used only when --decoding-method is modified_beam_search");
}

void OnlineRecognizerConfig::Validate() const {
  if (nn_model.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --nn-model";
  }
  AssertFileExists(nn_model);

  if (tokens.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --tokens";
  }
  AssertFileExists(tokens);

  if (decoding_method != "greedy_search" &&
      decoding_method != "modified_beam_search") {
    SHERPA_LOG(FATAL)
        << "Unsupported decoding method: " << decoding_method
        << ". Supported values are: greedy_search, modified_beam_search";
  }

  if (decoding_method == "modified_beam_search") {
    SHERPA_CHECK_GT(num_active_paths, 0);
  }
}

static OnlineRecognitionResult Convert(OnlineTransducerDecoderResult src,
                                       const SymbolTable &sym) {
  OnlineRecognitionResult r;
  std::string text;
  for (auto i : src.tokens) {
    text.append(sym[i]);
  }
  r.text = std::move(text);
  r.tokens = std::move(src.tokens);
  r.timestamps = std::move(src.timestamps);

  return r;
}

class OnlineRecognizer::OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerImpl(const OnlineRecognizerConfig &config)
      : config_(config), symbol_table_(config.tokens) {
    if (config.use_gpu) {
      device_ = torch::Device("cuda:0");
    }

    model_ = std::make_unique<OnlineConvEmformerTransducerModel>(
        config.nn_model, device_);

    decoder_ =
        std::make_unique<OnlineTransducerGreedySearchDecoder>(model_.get());
  }

  std::unique_ptr<OnlineStream> CreateStream() {
    auto s = std::make_unique<OnlineStream>(config_.feat_config.fbank_opts);

    auto r = decoder_->GetEmptyResult();
    s->SetResult(r);

    auto state = model_->GetEncoderInitStates();
    s->SetState(state);

    return s;
  }

  bool IsReady(OnlineStream *s) {
    // TODO(fangjun): Pass chunk_size to OnlineStream on creation
    int32_t chunk_size = model_->ChunkSize();
    return s->NumFramesReady() - s->GetNumProcessedFrames() >= chunk_size;
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) {
    torch::NoGradGuard no_grad;

    SHERPA_CHECK_GT(n, 0);

    auto device = model_->Device();
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    std::vector<torch::Tensor> all_features(n);
    std::vector<torch::IValue> all_states(n);
    std::vector<int32_t> all_processed_frames(n);
    std::vector<OnlineTransducerDecoderResult> all_results(n);
    for (int32_t i = 0; i != n; ++i) {
      OnlineStream *s = ss[i];

      SHERPA_CHECK(IsReady(s));
      int32_t num_processed_frames = s->GetNumProcessedFrames();

      std::vector<torch::Tensor> features_vec(chunk_size);
      for (int32_t k = 0; k != chunk_size; ++k) {
        features_vec[k] = s->GetFrame(num_processed_frames + k);
      }

      torch::Tensor features = torch::cat(features_vec, /*dim*/ 0);

      all_features[i] = std::move(features);
      all_states[i] = s->GetState();
      all_processed_frames[i] = num_processed_frames;
      all_results[i] = s->GetResult();
    }  // for (int32_t i = 0; i != n; ++i) {

    auto batched_features = torch::stack(all_features, /*dim*/ 0);
    batched_features = batched_features.to(device);

    torch::Tensor features_length =
        torch::full({n}, chunk_size, torch::kLong).to(device);

    torch::IValue stacked_states = model_->StackStates(all_states);
    torch::Tensor processed_frames =
        torch::tensor(all_processed_frames, torch::kLong).to(device);

    torch::Tensor encoder_out;
    torch::Tensor encoder_out_lens;
    torch::IValue next_states;

    std::tie(encoder_out, encoder_out_lens, next_states) = model_->RunEncoder(
        batched_features, features_length, processed_frames, stacked_states);

    decoder_->Decode(encoder_out, &all_results);

    std::vector<torch::IValue> unstacked_states =
        model_->UnStackStates(next_states);

    for (int32_t i = 0; i != n; ++i) {
      OnlineStream *s = ss[i];
      s->SetResult(all_results[i]);
      s->SetState(std::move(unstacked_states[i]));
      s->GetNumProcessedFrames() += chunk_shift;
    }
  }

  OnlineRecognitionResult GetResult(OnlineStream *s) const {
    auto r = s->GetResult();  // we use a copy here as we will change it below
    decoder_->StripLeadingBlanks(&r);
    return Convert(r, symbol_table_);
  }

 private:
  OnlineRecognizerConfig config_;
  torch::Device device_{"cpu"};
  std::unique_ptr<OnlineTransducerModel> model_;
  std::unique_ptr<OnlineTransducerDecoder> decoder_;
  SymbolTable symbol_table_;
};

OnlineRecognizer::OnlineRecognizer(const OnlineRecognizerConfig &config)
    : impl_(std::make_unique<OnlineRecognizerImpl>(config)) {}

OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() {
  return impl_->CreateStream();
}

bool OnlineRecognizer::IsReady(OnlineStream *s) { return impl_->IsReady(s); }

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) {
  torch::NoGradGuard no_grad;
  impl_->DecodeStreams(ss, n);
}

OnlineRecognitionResult OnlineRecognizer::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

}  // namespace sherpa
