// sherpa/cpp_api/offline-recognizer-ctc-impl.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-recognizer-impl.h"
#include "sherpa/csrc/offline-conformer-ctc-model.h"
#include "sherpa/csrc/offline-ctc-decoder.h"
#include "sherpa/csrc/offline-ctc-model.h"
#include "sherpa/csrc/offline-ctc-one-best-decoder.h"
#include "sherpa/csrc/offline-wav2vec2-ctc-model.h"
#include "sherpa/csrc/offline-wenet-conformer-ctc-model.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

static OfflineRecognitionResult Convert(OfflineCtcDecoderResult src,
                                        const SymbolTable &sym,
                                        bool insert_space = false) {
  OfflineRecognitionResult r;
  std::string text;
  for (auto i : src.tokens) {
    text.append(sym[i]);

    if (insert_space) {
      text.append(" ");
    }
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
    } else if (class_name == "Wav2Vec2Model") {
      // This one is from torchaudio
      // https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/model.py#L11
      model_ =
          std::make_unique<OfflineWav2Vec2CtcModel>(config.nn_model, device_);
      return_waveform_ = true;
      symbol_table_.Replace(symbol_table_["|"], " ", "|");
    } else {
      std::string s =
          "Support only models from icefall, wenet and torchaudio\n"
          "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/"
          "conformer_ctc/conformer.py#L27"
          "\n"
          "https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/"
          "asr_model.py#L42"
          "\n"
          "https://github.com/pytorch/audio/blob/main/torchaudio/models/"
          "wav2vec2/model.py#L11"
          "\n";

      TORCH_CHECK(false, s);
    }

    decoder_ = std::make_unique<OfflineCtcOneBestDecoder>(
        config.ctc_decoder_config, device_);

    // If we provide HLG, the decoder will return word IDs, we need
    // to insert a space between each word.
    insert_space_ = !config.ctc_decoder_config.hlg.empty();
  }

  std::unique_ptr<OfflineStream> CreateStream() override {
    return std::make_unique<OfflineStream>(&fbank_, return_waveform_,
                                           normalize_samples_);
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

    // If return_waveform is false, features_vec contains 2-D tensors of shape
    // (num_frames, feature_dim). In this case, we should use the padding value
    // -23.
    //
    // If return_waveform is true, features_vec contains 1-D tensors of shape
    // (num_samples,). In this case, we use 0 as the padding value.
    auto features = torch::nn::utils::rnn::pad_sequence(
        features_vec, /*batch_first*/ true,
        /*padding_value*/ return_waveform_ ? 0 : -23.025850929940457f);

    auto features_length = torch::tensor(features_length_vec);

    torch::IValue ivalue = model_->Forward(features, features_length);
    torch::Tensor log_prob = model_->GetLogSoftmaxOut(ivalue);
    torch::Tensor log_prob_len = model_->GetLogSoftmaxOutLength(ivalue);

    auto results =
        decoder_->Decode(log_prob, log_prob_len, model_->SubsamplingFactor());
    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(Convert(results[i], symbol_table_, insert_space_));
    }
  }

 private:
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCtcModel> model_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
  kaldifeat::Fbank fbank_;
  torch::Device device_;
  bool normalize_samples_ = true;
  bool return_waveform_ = false;
  bool insert_space_ = false;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_
