// sherpa/cpp_api/offline-recognizer-sense-voice-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/offline-ctc-decoder.h"
#include "sherpa/csrc/offline-ctc-greedy-search-decoder.h"
#include "sherpa/csrc/offline-sense-voice-model.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

static OfflineRecognitionResult ConvertSenseVoice(
    const OfflineCtcDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;
  int32_t k = 0;
  for (auto i : src.tokens) {
    k += 1;
    if (k <= 4) {
      // skip <|en|><|NEUTRAL|><|Speech|><|woitn|>
      continue;
    }
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

    decoder_ = std::make_unique<OfflineCtcGreedySearchDecoder>();

    WarmUp();
  }

  std::unique_ptr<OfflineStream> CreateStream() override {
    return std::make_unique<OfflineStream>(&fbank_, config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) override {
    InferenceMode no_grad;

    std::vector<torch::Tensor> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      auto f = ss[i]->GetFeatures();
      f = ApplyLFR(f);
      f = ApplyCMVN(f);
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

    const auto &meta_data = model_->GetModelMetadata();
    int32_t language_id = meta_data.lang2id.at("auto");
    if (meta_data.lang2id.count(config_.model.sense_voice.language)) {
      language_id = meta_data.lang2id.at(config_.model.sense_voice.language);
    }
    std::vector<int32_t> language(n, language_id);

    std::vector<int32_t> use_itn(n, config_.model.sense_voice.use_itn
                                        ? meta_data.with_itn_id
                                        : meta_data.without_itn_id);

    auto language_tensor = torch::tensor(language, torch::kInt).to(device);
    auto use_itn_tensor = torch::tensor(use_itn, torch::kInt).to(device);

    auto outputs = model_->RunForward(features, features_length,
                                      language_tensor, use_itn_tensor);

    auto logits = outputs.first;
    auto logits_length = outputs.second;

    auto results = decoder_->Decode(logits, logits_length);

    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(ConvertSenseVoice(
          results[i], symbol_table_,
          config_.feat_config.fbank_opts.frame_opts.frame_shift_ms,
          meta_data.window_shift));
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
    features = ApplyLFR(features);
    features = ApplyCMVN(features);
    auto features_length = torch::tensor({features.size(0)});
    features = features.unsqueeze(0);

    auto device = model_->Device();

    features = features.to(device);
    features_length = features_length.to(device);

    const auto &meta_data = model_->GetModelMetadata();
    int32_t language_id = meta_data.lang2id.at("auto");

    std::vector<int32_t> language(1, language_id);

    std::vector<int32_t> use_itn(1, config_.model.sense_voice.use_itn
                                        ? meta_data.with_itn_id
                                        : meta_data.without_itn_id);

    auto language_tensor = torch::tensor(language, torch::kInt).to(device);
    auto use_itn_tensor = torch::tensor(use_itn, torch::kInt).to(device);

    auto outputs = model_->RunForward(features, features_length,
                                      language_tensor, use_itn_tensor);

    SHERPA_LOG(INFO) << "WarmUp ended";
  }

  torch::Tensor ApplyLFR(torch::Tensor features) const {
    const auto &meta_data = model_->GetModelMetadata();

    int32_t lfr_window_size = meta_data.window_size;
    int32_t lfr_window_shift = meta_data.window_shift;

    int32_t num_frames = features.size(0);
    int32_t feat_dim = features.size(1);

    int32_t new_num_frames =
        (num_frames - lfr_window_size) / lfr_window_shift + 1;

    int32_t new_feat_dim = feat_dim * lfr_window_size;

    return features
        .as_strided({new_num_frames, new_feat_dim},
                    {lfr_window_shift * feat_dim, 1})
        .clone();
  }

  torch::Tensor ApplyCMVN(torch::Tensor features) const {
    const auto &meta_data = model_->GetModelMetadata();

    return (features + meta_data.neg_mean) * meta_data.inv_stddev;
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
