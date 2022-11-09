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
        device_(torch::kCPU) {
    if (config.use_gpu) {
      device_ = torch::Device("cuda:0");
    }
    model_ =
        std::make_unique<OfflineConformerCtcModel>(config.nn_model, device_);

    decoder_ = std::make_unique<OfflineCtcOneBestDecoder>(
        config.ctc_decoder_config, device_);
  }

  std::unique_ptr<OfflineStream> CreateStream() override {
    return std::make_unique<OfflineStream>(&fbank_);
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
    std::cerr << "features.shape: " << features.sizes() << "\n";
    std::cerr << "features_length: " << features_length << "\n";

    torch::IValue ivalue = model_->Forward(features, features_length);
    torch::Tensor log_prob = model_->GetLogSoftmaxOut(ivalue);
    torch::Tensor log_prob_len = model_->GetLogSoftmaxOutLength(ivalue);

    std::cerr << "log_prob.shape: " << log_prob.sizes() << "\n";
    std::cerr << "log_prob_len: " << log_prob_len << "\n";

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
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_
