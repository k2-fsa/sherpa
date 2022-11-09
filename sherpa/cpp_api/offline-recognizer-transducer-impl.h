// sherpa/cpp_api/offline-recognizer-transducer-impl.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-recognizer-impl.h"
#include "sherpa/csrc/offline-conformer-transducer-model.h"
#include "sherpa/csrc/offline-transducer-decoder.h"
#include "sherpa/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa/csrc/offline-transducer-model.h"
#include "sherpa/csrc/offline-transducer-modified-beam-search-decoder.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

static OfflineRecognitionResult Convert(OfflineTransducerDecoderResult src,
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

class OfflineRecognizerTransducerImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerImpl(
      const OfflineRecognizerConfig &config)
      : symbol_table_(config.tokens),
        fbank_(config.feat_config.fbank_opts),
        device_(torch::kCPU) {
    if (config.use_gpu) {
      device_ = torch::Device("cuda:0");
    }
    model_ = std::make_unique<OfflineConformerTransducerModel>(config.nn_model,
                                                               device_);

    if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineTransducerGreedySearchDecoder>(model_.get());
    } else if (config.decoding_method == "modified_beam_search") {
      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), config.num_active_paths);
    } else {
      TORCH_CHECK(false,
                  "Unsupported decoding method: ", config.decoding_method);
    }
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

    torch::Tensor encoder_out;
    torch::Tensor encoder_out_length;

    std::tie(encoder_out, encoder_out_length) =
        model_->RunEncoder(features, features_length);
    encoder_out_length = encoder_out_length.cpu();

    auto results = decoder_->Decode(encoder_out, encoder_out_length);
    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(Convert(results[i], symbol_table_));
    }
  }

 private:
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineTransducerModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
  kaldifeat::Fbank fbank_;
  torch::Device device_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_TRANSDUCER_IMPL_H_