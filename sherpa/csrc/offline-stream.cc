// sherpa/cpp_api/offline-stream.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/cpp_api/offline-stream.h"

#include <memory>
#include <string>

#include "nlohmann/json.hpp"
#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/csrc/fbank-features.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

std::string OfflineRecognitionResult::AsJsonString() const {
  using json = nlohmann::json;
  json j;
  j["text"] = text;
  j["tokens"] = tokens;

  std::ostringstream os;
  os << "[";
  std::string sep = "";
  for (auto t : timestamps) {
    os << sep << std::fixed << std::setprecision(2) << t;
    sep = ",";
  }
  os << "]";

  // NOTE: We don't use j["timestamps"] = timestamps;
  // because we need to control the number of decimal points to keep
  j["timestamps"] = os.str();

  return j.dump();
}

class OfflineStream::OfflineStreamImpl {
 public:
  OfflineStreamImpl(kaldifeat::Fbank *fbank, const FeatureConfig &feat_config,
                    ContextGraphPtr context_graph)
      : fbank_(fbank),
        feat_config_(feat_config),
        context_graph_(context_graph) {
    if (!feat_config_.nemo_normalize.empty()) {
      SHERPA_CHECK_EQ(feat_config_.nemo_normalize, "per_feature")
          << "Only per_feature is implemented at present";
    }
  }

  void AcceptWaveFile(const std::string &wave_file) {
    torch::Tensor samples =
        ReadWave(wave_file, fbank_->GetFrameOptions().samp_freq).first;
    if (!feat_config_.normalize_samples) {
      samples.mul_(32767);
    }

    if (feat_config_.return_waveform) {
      // We return audio samples directly, e.g., for Wav2Vec2.0
      features_ = samples;
    } else {
      features_ = ComputeFeatures(*fbank_, {samples})[0];
      features_ = Normalize(features_);
    }
  }

  void AcceptSamples(const float *samples, int32_t n) {
    torch::Tensor tensor =
        torch::from_blob(const_cast<float *>(samples), {n}, torch::kFloat);

    if (!feat_config_.normalize_samples) {
      tensor.mul_(32767);
    }

    if (feat_config_.return_waveform) {
      // We return audio samples directly, e.g., for Wav2Vec2.0
      features_ = tensor.clone();
    } else {
      features_ = ComputeFeatures(*fbank_, {tensor})[0];
      features_ = Normalize(features_);
    }
  }

  void AcceptFeatures(const float *features, int32_t num_frames,
                      int32_t num_channels) {
    features_ = torch::from_blob(const_cast<float *>(features),
                                 {num_frames, num_channels}, torch::kFloat)
                    .clone();
  }

  const torch::Tensor &GetFeatures() const { return features_; }

  void SetResult(const OfflineRecognitionResult &r) { result_ = r; }

  const OfflineRecognitionResult &GetResult() const { return result_; }

  const ContextGraphPtr &GetContextGraph() const { return context_graph_; }

 private:
  torch::Tensor Normalize(torch::Tensor features) const {
    if (feat_config_.nemo_normalize.empty()) {
      return features;
    }

    if (feat_config_.nemo_normalize == "per_feature") {
      torch::Tensor mean = features.mean(0 /*dim*/, true /*keepdim*/);
      torch::Tensor std = features.std(0 /*dim*/, true /*keepdim*/);

      return (features - mean) / (std + 1e-5f);
    }

    SHERPA_LOG(FATAL) << "Unsupported nemo_normalize: "
                      << feat_config_.nemo_normalize;
    return {};  // unreachable code; to make the compiler happy
  }

 private:
  torch::Tensor features_;
  OfflineRecognitionResult result_;
  kaldifeat::Fbank *fbank_ = nullptr;  // not owned
  FeatureConfig feat_config_;
  ContextGraphPtr context_graph_;
};

OfflineStream::~OfflineStream() = default;

OfflineStream::OfflineStream(kaldifeat::Fbank *fbank,
                             const FeatureConfig &feat_config,
                             ContextGraphPtr context_graph /* nullptr */)
    : impl_(std::make_unique<OfflineStreamImpl>(fbank, feat_config,
                                                context_graph)) {}

void OfflineStream::AcceptWaveFile(const std::string &filename) {
  impl_->AcceptWaveFile(filename);
}

void OfflineStream::AcceptSamples(const float *samples, int32_t n) {
  impl_->AcceptSamples(samples, n);
}

void OfflineStream::AcceptFeatures(const float *features, int32_t num_frames,
                                   int32_t num_channels) {
  impl_->AcceptFeatures(features, num_frames, num_channels);
}

const torch::Tensor &OfflineStream::GetFeatures() const {
  return impl_->GetFeatures();
}

const ContextGraphPtr &OfflineStream::GetContextGraph() const {
  return impl_->GetContextGraph();
}

/** Set the recognition result for this stream. */
void OfflineStream::SetResult(const OfflineRecognitionResult &r) {
  impl_->SetResult(r);
}

/** Get the recognition result of this stream */
const OfflineRecognitionResult &OfflineStream::GetResult() const {
  return impl_->GetResult();
}

}  // namespace sherpa
