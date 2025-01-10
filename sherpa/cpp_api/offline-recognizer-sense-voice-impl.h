// sherpa/cpp_api/offline-recognizer-sense-voice-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/offline-sense-voice-model.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

class OfflineRecognizerSenseVoiceImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerSenseVoiceImpl(
      const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config.model.tokens),
        fbank_(config.feat_config.fbank_opts) {
    config.ctc_decoder_config.Validate();

    model_ = std::make_unique<OfflineSenseVoiceModel>(config.model);

    config_.feat_config.normalize_samples =
        model_->GetModelMetadata().normalize_samples;
  }

  std::unique_ptr<OfflineStream> CreateStream() override {
    return std::make_unique<OfflineStream>(&fbank_, config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) override {
    InferenceMode no_grad;

    std::vector<torch::Tensor> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      const auto &f = ss[i]->GetFeatures();
      features_vec[i] = f;
      features_length_vec[i] = f.size(0);
    }

    auto device = model_->Device();

    // If return_waveform is true, features_vec contains 1-D tensors of shape
    // (num_samples,). In this case, we use 0 as the padding value.
    auto features =
        torch::nn::utils::rnn::pad_sequence(features_vec, /*batch_first*/ true,
                                            /*padding_value*/ 0)
            .to(device);

    auto features_length = torch::tensor(features_length_vec).to(device);

    /*
    {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
    self.textnorm_dict = {"withitn": 14, "woitn": 15}
    */
    std::vector<int32_t> language(n, 0);
    std::vector<int32_t> use_itn(n,
                                 config_.model.sense_voice.use_itn ? 14 : 15);
    auto language_tensor = torch::tensor(language, torch::kInt).to(device);
    auto use_itn_tensor = torch::tensor(use_itn, torch::kInt).to(device);
    /*
        auto outputs = model
                           .run_method("forward", features, features_length,
                                       language_tensor, use_itn_tensor)
                           .toTuple();

        auto logits = outputs->elements()[0];
        auto logits_length = outputs->elements()[1].toTensor();
    */
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  kaldifeat::Fbank fbank_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
  std::unique_ptr<OfflineSenseVoiceModel> model_;
};
}  // namespace sherpa
#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
