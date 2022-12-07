// sherpa/cpp_api/offline-recognizer-transducer-impl.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/autocast.h"
#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-recognizer-impl.h"
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
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);

    r.tokens.push_back(std::move(sym));
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
          model_.get(), config.num_active_paths);
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
    bool return_waveform = false;
    return std::make_unique<OfflineStream>(
        &fbank_, return_waveform, config_.feat_config.normalize_samples);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) override {
    torch::NoGradGuard no_grad;

    std::vector<torch::Tensor> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
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

    {
      // Note: We only use AMP for running the encoder.
      AutoCast autocast(config_.use_amp, config_.use_gpu);
      std::tie(encoder_out, encoder_out_length) =
          model_->RunEncoder(features, features_length);
    }
    encoder_out = encoder_out.to(torch::kFloat);
    encoder_out_length = encoder_out_length.cpu();

    auto results = decoder_->Decode(encoder_out, encoder_out_length);
    for (int32_t i = 0; i != n; ++i) {
      auto ans =
          Convert(results[i], symbol_table_,
                  config_.feat_config.fbank_opts.frame_opts.frame_shift_ms,
                  model_->SubsamplingFactor());

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

    features = features.unsqueeze(0).to(device_);
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
