// sherpa/cpp_api/offline-stream.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/cpp_api/offline-stream.h"

#include <memory>
#include <string>

#include "nlohmann/json.hpp"
#include "sherpa/csrc/fbank-features.h"

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
  OfflineStreamImpl(kaldifeat::Fbank *fbank, bool return_waveform,
                    bool normalize_samples)
      : fbank_(fbank),
        return_waveform_(return_waveform),
        normalize_samples_(normalize_samples) {}

  void AcceptWaveFile(const std::string &wave_file) {
    torch::Tensor samples =
        ReadWave(wave_file, fbank_->GetFrameOptions().samp_freq).first;
    if (!normalize_samples_) {
      samples.mul_(32767);
    }

    if (return_waveform_) {
      // We return audio samples directly, e.g., for Wav2Vec2.0
      features_ = samples;
    } else {
      features_ = ComputeFeatures(*fbank_, {samples})[0];
    }
  }

  void AcceptSamples(const float *samples, int32_t n) {
    torch::Tensor tensor =
        torch::from_blob(const_cast<float *>(samples), {n}, torch::kFloat);

    if (!normalize_samples_) {
      tensor.mul_(32767);
    }

    if (return_waveform_) {
      features_ = tensor.clone();
    } else {
      // We return audio samples directly, e.g., for Wav2Vec2.0
      features_ = ComputeFeatures(*fbank_, {tensor})[0];
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

 private:
  torch::Tensor features_;
  OfflineRecognitionResult result_;
  kaldifeat::Fbank *fbank_ = nullptr;  // not owned
  bool return_waveform_ = false;
  bool normalize_samples_ = true;
};

OfflineStream::~OfflineStream() = default;

OfflineStream::OfflineStream(kaldifeat::Fbank *fbank,
                             bool return_waveform /*= false*/,
                             bool normalize_samples /*= true*/)
    : impl_(std::make_unique<OfflineStreamImpl>(fbank, return_waveform,
                                                normalize_samples)) {}

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

/** Set the recognition result for this stream. */
void OfflineStream::SetResult(const OfflineRecognitionResult &r) {
  impl_->SetResult(r);
}

/** Get the recognition result of this stream */
const OfflineRecognitionResult &OfflineStream::GetResult() const {
  return impl_->GetResult();
}

}  // namespace sherpa
