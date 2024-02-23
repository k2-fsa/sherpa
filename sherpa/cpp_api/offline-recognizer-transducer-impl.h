// sherpa/cpp_api/offline-recognizer-transducer-impl.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-recognizer-impl.h"
#include "sherpa/csrc/byte_util.h"
#include "sherpa/csrc/context-graph.h"
#include "sherpa/csrc/offline-conformer-transducer-model.h"
#include "sherpa/csrc/offline-transducer-decoder.h"
#include "sherpa/csrc/offline-transducer-fast-beam-search-decoder.h"
#include "sherpa/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa/csrc/offline-transducer-model.h"
#include "sherpa/csrc/offline-transducer-modified-beam-search-decoder.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

static OfflineRecognitionResult Convert(
    const OfflineTransducerDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor, bool use_bbpe) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);

    r.tokens.push_back(std::move(sym));
  }

  if (use_bbpe) {
    auto bu = GetByteUtil();
    text = bu->Decode(text);
  }

  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  return r;
}

class OfflineRecognizerTransducerImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerImpl(
      const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config.tokens),
        fbank_(config.feat_config.fbank_opts),
        device_(torch::kCPU) {
    if (config.use_gpu) {
      device_ = torch::Device("cuda:0");
    }
    model_ = std::make_unique<OfflineConformerTransducerModel>(config.nn_model,
                                                               device_);

    WarmUp();

    if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineTransducerGreedySearchDecoder>(model_.get());
    } else if (config.decoding_method == "modified_beam_search") {
      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), config.num_active_paths, config.temperature);
    } else if (config.decoding_method == "fast_beam_search") {
      config.fast_beam_search_config.Validate();

      decoder_ = std::make_unique<OfflineTransducerFastBeamSearchDecoder>(
          model_.get(), config.fast_beam_search_config);
    } else {
      TORCH_CHECK(false,
                  "Unsupported decoding method: ", config.decoding_method);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() override {
    return std::make_unique<OfflineStream>(&fbank_, config_.feat_config);
  }

  std::unique_ptr<OfflineStream> CreateStream(
      const std::vector<std::vector<int32_t>> &context_list) override {
    // We create context_graph at this level, because we might have default
    // context_graph(will be added later if needed) that belongs to the whole
    // model rather than each stream.
    auto context_graph =
        std::make_shared<ContextGraph>(context_list, config_.context_score);
    return std::make_unique<OfflineStream>(&fbank_, config_.feat_config,
                                           context_graph);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) override {
    InferenceMode no_grad;

    bool has_context_graph = false;
    std::vector<torch::Tensor> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      if (!has_context_graph && ss[i]->GetContextGraph())
        has_context_graph = true;
      const auto &f = ss[i]->GetFeatures();
      features_vec[i] = f;
      features_length_vec[i] = f.size(0);
    }

    auto features = torch::nn::utils::rnn::pad_sequence(
                        features_vec, /*batch_first*/ true,
                        /*padding_value*/ -23.025850929940457f)
                        .to(device_);

    auto features_length = torch::tensor(features_length_vec).to(device_);

    torch::Tensor encoder_out;
    torch::Tensor encoder_out_length;

    std::tie(encoder_out, encoder_out_length) =
        model_->RunEncoder(features, features_length);
    encoder_out_length = encoder_out_length.cpu();

    OfflineStream **streams = has_context_graph ? ss : nullptr;
    int32_t num_streams = has_context_graph ? n : 0;
    auto results =
        decoder_->Decode(encoder_out, encoder_out_length, streams, num_streams);

    for (int32_t i = 0; i != n; ++i) {
      auto ans =
          Convert(results[i], symbol_table_,
                  config_.feat_config.fbank_opts.frame_opts.frame_shift_ms,
                  model_->SubsamplingFactor(), config_.use_bbpe);

      ss[i]->SetResult(ans);
    }
  }

 private:
  void WarmUp() {
    SHERPA_LOG(INFO) << "WarmUp begins";
    auto s = CreateStream();
    float sample_rate = fbank_.GetFrameOptions().samp_freq;
    std::vector<float> samples(2 * sample_rate, 0);
    s->AcceptSamples(samples.data(), samples.size());
    auto features = s->GetFeatures();
    auto features_length = torch::tensor({features.size(0)});
    features = features.unsqueeze(0);

    features = features.to(device_);
    features_length = features_length.to(device_);

    model_->WarmUp(features, features_length);
    SHERPA_LOG(INFO) << "WarmUp ended";
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineTransducerModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
  kaldifeat::Fbank fbank_;
  torch::Device device_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
