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

#include "sherpa/cpp_api/online-recognizer.h"

#include <utility>

#include "nlohmann/json.hpp"
#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online-conformer-transducer-model.h"
#include "sherpa/csrc/online-conv-emformer-transducer-model.h"
#include "sherpa/csrc/online-emformer-transducer-model.h"
#include "sherpa/csrc/online-lstm-transducer-model.h"
#include "sherpa/csrc/online-transducer-decoder.h"
#include "sherpa/csrc/online-transducer-greedy-search-decoder.h"
#include "sherpa/csrc/online-transducer-model.h"
#include "sherpa/csrc/online-transducer-modified-beam-search-decoder.h"
#include "sherpa/csrc/symbol-table.h"

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

  po->Register("encoder-model", &encoder_model,
               "Path to the encoder model for OnlineLstmTransducerModel.");

  po->Register("decoder-model", &decoder_model,
               "Path to the decoder model for OnlineLstmTransducerModel.");

  po->Register("joiner-model", &joiner_model,
               "Path to the joiner model for OnlineLstmTransducerModel.");

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

  po->Register("decode-left-context", &left_context,
               "Used only for streaming Conformer, i.e, models from "
               "pruned_transducer_statelessX in icefall. "
               "Number of frames after subsampling during decoding.");

  po->Register("decode-right-context", &right_context,
               "Used only for streaming Conformer, i.e, models from "
               "pruned_transducer_statelessX in icefall. "
               "Number of frames after subsampling during decoding.");

  po->Register("decode-chunk-size", &chunk_size,
               "Used only for streaming Conformer, i.e, models from "
               "pruned_transducer_statelessX in icefall. "
               "Number of frames after subsampling during decoding.");
}

void OnlineRecognizerConfig::Validate() const {
  if (!nn_model.empty()) {
    SHERPA_CHECK_EQ(encoder_model.empty(), true);
    SHERPA_CHECK_EQ(decoder_model.empty(), true);
    SHERPA_CHECK_EQ(joiner_model.empty(), true);

    AssertFileExists(nn_model);
  } else {
    SHERPA_CHECK_EQ(encoder_model.empty(), false)
        << "If you don't provide --nn-model, please provide --encoder_model "
           "instead";
    SHERPA_CHECK_EQ(decoder_model.empty(), false);
    SHERPA_CHECK_EQ(joiner_model.empty(), false);

    AssertFileExists(decoder_model);
    AssertFileExists(decoder_model);
    AssertFileExists(joiner_model);
  }

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

    if (config.nn_model.empty()) {
      // For OnlineLstmTransducerModel
      model_ = std::make_unique<OnlineLstmTransducerModel>(
          config.encoder_model, config.decoder_model, config.joiner_model,
          device_);
    } else {
      torch::jit::Module m = torch::jit::load(config.nn_model, torch::kCPU);
      auto encoder = m.attr("encoder").toModule();
      std::string class_name = encoder.type()->name()->name();

      if (class_name == "Emformer") {
        if (encoder.find_method("infer")) {
          // Emformer from torchaudio
          model_ = std::make_unique<OnlineEmformerTransducerModel>(
              config.nn_model, device_);
        } else {
          // ConvEmformer from icefall
          model_ = std::make_unique<OnlineConvEmformerTransducerModel>(
              config.nn_model, device_);
        }
      } else if (class_name == "Conformer") {
        int32_t left_context = config.left_context;
        int32_t right_context = config.right_context;
        int32_t chunk_size = config.chunk_size;
        SHERPA_CHECK_GT(left_context, 0);
        SHERPA_CHECK_GT(right_context, 0);
        SHERPA_CHECK_GT(chunk_size, 0);

        model_ = std::make_unique<OnlineConformerTransducerModel>(
            config.nn_model, left_context, right_context, chunk_size, device_);
      } else {
        std::string s =
            "Support only the following models from icefall:\n"
            "conv_emformer_transducer_stateless2\n"
            "pruned_stateless_emformer_rnnt2\n"
            "pruned_transducer_stateless{2,3,4,5}\n";
        TORCH_CHECK(false, s);
      }
    }

    if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OnlineTransducerGreedySearchDecoder>(model_.get());
    } else if (config.decoding_method == "modified_beam_search") {
      decoder_ = std::make_unique<OnlineTransducerModifiedBeamSearchDecoder>(
          model_.get(), config.num_active_paths);
    } else {
      TORCH_CHECK(false,
                  "Unsupported decoding method: ", config.decoding_method);
    }
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
