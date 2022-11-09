// sherpa/cpp_api/offline-recognizer-ctc-impl.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-recognizer-impl.h"
#include "sherpa/csrc/offline-conformer-ctc-model.h"
#include "sherpa/csrc/offline-ctc-decoder.h"
#include "sherpa/csrc/offline-ctc-model.h"
#include "sherpa/csrc/offline-ctc-one-best-decoder.h"
#include "sherpa/csrc/offline-wenet-conformer-ctc-model.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

static OfflineRecognitionResult Convert(OfflineCtcDecoderResult src,
                                        const SymbolTable &sym) {
  OfflineRecognitionResult r;
  std::string text;
  for (auto i : src.tokens) {
    text.append(sym[i]);
  }
  r.text = std::move(text);
  r.tokens = std::move(src.tokens);
  r.timestamps = std::move(src.timestamps);

  return r;
}

class OfflineRecognizerCtcImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerCtcImpl(const OfflineRecognizerConfig &config)
      : symbol_table_(config.tokens),
        fbank_(config.feat_config.fbank_opts),
        device_(torch::kCPU),
        normalize_samples_(config.feat_config.normalize_samples) {
    if (config.use_gpu) {
      device_ = torch::Device("cuda:0");
    }

    torch::jit::Module m = torch::jit::load(config.nn_model, torch::kCPU);
    // We currently support: icefall, wenet, torchaudio.
    std::string class_name = m.type()->name()->name();
    if (class_name == "ASRModel") {
      // this one is from wenet, see
      // https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/asr_model.py#L42
      model_ = std::make_unique<OfflineWenetConformerCtcModel>(config.nn_model,
                                                               device_);
    } else if (class_name == "Conformer") {
      // this one is from icefall, see
      // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/conformer.py#L27
      model_ =
          std::make_unique<OfflineConformerCtcModel>(config.nn_model, device_);
    } else {
      std::string s =
          "Support only the models from icefall, wenet and torchaudio\n"
          "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/"
          "conformer_ctc/conformer.py#L27"
          "\n"
          "https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/"
          "asr_model.py#L42"
          "\n";

      TORCH_CHECK(false, s);
    }

    decoder_ = std::make_unique<OfflineCtcOneBestDecoder>(
        config.ctc_decoder_config, device_);
  }

  std::unique_ptr<OfflineStream> CreateStream() override {
    return std::make_unique<OfflineStream>(&fbank_, normalize_samples_);
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
        /*padding_value*/ -23.025850929940457f);

    auto features_length = torch::tensor(features_length_vec);

    torch::IValue ivalue = model_->Forward(features, features_length);
    torch::Tensor log_prob = model_->GetLogSoftmaxOut(ivalue);
    torch::Tensor log_prob_len = model_->GetLogSoftmaxOutLength(ivalue);
    std::cerr << "log_prob.sizes() " << log_prob.sizes() << "\n";
    std::cerr << "log_prob_len " << log_prob_len << "\n";

    auto results =
        decoder_->Decode(log_prob, log_prob_len, model_->SubsamplingFactor());
    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(Convert(results[i], symbol_table_));
    }
  }

 private:
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCtcModel> model_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
  kaldifeat::Fbank fbank_;
  torch::Device device_;
  bool normalize_samples_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_
