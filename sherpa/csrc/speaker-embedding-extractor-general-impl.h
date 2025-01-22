// sherpa/csrc/speaker-embedding-extractor-general-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#define SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/macros.h"
#include "sherpa/cpp_api/offline-stream.h"
#include "sherpa/csrc/speaker-embedding-extractor-impl.h"
#include "sherpa/csrc/speaker-embedding-extractor-model.h"

namespace sherpa {

class SpeakerEmbeddingExtractorGeneralImpl
    : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorGeneralImpl(
      const SpeakerEmbeddingExtractorConfig &config)
      : model_(config) {
    // TODO(fangjun): make it configurable
    feat_config_.fbank_opts.frame_opts.dither = 0;
    feat_config_.fbank_opts.frame_opts.snip_edges = true;
    feat_config_.fbank_opts.frame_opts.samp_freq = 16000;
    feat_config_.fbank_opts.mel_opts.num_bins = 80;
    feat_config_.normalize_samples = true;

    fbank_ = std::make_unique<kaldifeat::Fbank>(feat_config_.fbank_opts);

    WarmUp();
  }

  int32_t Dim() const override { return model_.GetModelMetadata().output_dim; }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(fbank_.get(), feat_config_);
  }

  torch::Tensor Compute(OfflineStream *s) const override {
    InferenceMode no_grad;
    auto features = s->GetFeatures();
    features -= features.mean(0, true);
    features = features.unsqueeze(0);
    auto device = model_.Device();
    return model_.Compute(features.to(device));
  }

  torch::Tensor Compute(OfflineStream **ss, int32_t n) const override {
    InferenceMode no_grad;
    if (n == 1) {
      return Compute(ss[0]);
    }

    std::vector<torch::Tensor> features_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      auto f = ss[i]->GetFeatures();
      f -= f.mean(0, true);
      features_vec[i] = f;
    }

    auto device = model_.Device();

    auto features =
        torch::nn::utils::rnn::pad_sequence(features_vec, true, 0).to(device);

    return model_.Compute(features);
  }

 private:
  void WarmUp() {
    InferenceMode no_grad;
    SHERPA_LOG(INFO) << "WarmUp begins";
    auto s = CreateStream();
    float sample_rate = fbank_->GetFrameOptions().samp_freq;
    std::vector<float> samples(2 * sample_rate, 0);
    s->AcceptSamples(samples.data(), samples.size());

    auto embedding = Compute(s.get());

    model_.GetModelMetadata().output_dim = embedding.size(1);

    SHERPA_LOG(INFO) << "WarmUp ended";
  }

 private:
  SpeakerEmbeddingExtractorModel model_;
  std::unique_ptr<kaldifeat::Fbank> fbank_;
  FeatureConfig feat_config_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
