// sherpa/cpp_api/offline-recognizer-ctc-impl.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/autocast.h"
#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-recognizer-impl.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/offline-conformer-ctc-model.h"
#include "sherpa/csrc/offline-ctc-decoder.h"
#include "sherpa/csrc/offline-ctc-model.h"
#include "sherpa/csrc/offline-ctc-one-best-decoder.h"
#include "sherpa/csrc/offline-nemo-enc-dec-ctc-model-bpe.h"
#include "sherpa/csrc/offline-wav2vec2-ctc-model.h"
#include "sherpa/csrc/offline-wenet-conformer-ctc-model.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

static OfflineRecognitionResult Convert(const OfflineCtcDecoderResult &src,
                                        const SymbolTable &sym_table,
                                        int32_t frame_shift_ms,
                                        int32_t subsampling_factor) {
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

class OfflineRecognizerCtcImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerCtcImpl(const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config.tokens),
        fbank_(config.feat_config.fbank_opts),
        device_(torch::kCPU) {
    config.ctc_decoder_config.Validate();

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
      config_.feat_config.return_waveform = true;
      symbol_table_.Replace(symbol_table_["|"], " ", "|");
      // See Section 4.2 of
      // https://arxiv.org/pdf/2006.11477.pdf
      config_.feat_config.fbank_opts.frame_opts.frame_shift_ms = 20;
      SHERPA_LOG(WARNING) << "Set frame_shift_ms to 20 for wav2vec 2.0";
    } else if (class_name == "EncDecCTCModelBPE") {
      // This one is from NeMo
      // See
      // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_bpe_models.py#L34
      //
      model_ = std::make_unique<OfflineNeMoEncDecCTCModelBPE>(config.nn_model,
                                                              device_);
    } else if (class_name == "EncDecCTCModel") {
      // This one is from NeMo
      // See
      // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py#L41
      //
      model_ =
          std::make_unique<OfflineNeMoEncDecCTCModel>(config.nn_model, device_);
    } else {
      std::ostringstream os;
      os << "Support only models from icefall, wenet, torchaudio, and NeMo\n"
            "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/"
            "ASR/"
            "conformer_ctc/conformer.py#L27"
            "\n"
            "https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/"
            "asr_model.py#L42"
            "\n"
            "https://github.com/pytorch/audio/blob/main/torchaudio/models/"
            "wav2vec2/model.py#L11"
            "\n"
            "https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/"
            "models/ctc_bpe_models.py#L34"
            "\n"
            "https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/"
            "models/ctc_models.py#L41"
         << "\n"
         << "Given: " << class_name << "\n";

      TORCH_CHECK(false, os.str());
    }

    WarmUp();

    decoder_ = std::make_unique<OfflineCtcOneBestDecoder>(
        config.ctc_decoder_config, device_, model_->VocabSize());
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

    // If return_waveform is false, features_vec contains 2-D tensors of shape
    // (num_frames, feature_dim). In this case, we should use the padding
    // value -23.
    //
    // If return_waveform is true, features_vec contains 1-D tensors of shape
    // (num_samples,). In this case, we use 0 as the padding value.
    auto features = torch::nn::utils::rnn::pad_sequence(
        features_vec, /*batch_first*/ true,
        /*padding_value*/ return_waveform_ ? 0 : -23.025850929940457f);

    auto features_length = torch::tensor(features_length_vec);

    torch::IValue ivalue;
    {
      AutoCast autocast(config_.use_amp, config_.use_gpu);
      ivalue = model_->Forward(features, features_length);
    }

    torch::Tensor log_prob = model_->GetLogSoftmaxOut(ivalue).to(torch::kFloat);
    torch::Tensor log_prob_len = model_->GetLogSoftmaxOutLength(ivalue);
    if (!log_prob_len.defined()) {
      log_prob_len =
          torch::floor_divide(features_length, model_->SubsamplingFactor());
      log_prob_len = log_prob_len.to(log_prob.device());
    }

    auto results =
        decoder_->Decode(log_prob, log_prob_len, model_->SubsamplingFactor());
    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(
          Convert(results[i], symbol_table_,
                  config_.feat_config.fbank_opts.frame_opts.frame_shift_ms,
                  model_->SubsamplingFactor()));
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
  std::unique_ptr<OfflineCtcModel> model_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
  kaldifeat::Fbank fbank_;
  torch::Device device_;
  bool return_waveform_ = false;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_CTC_IMPL_H_
