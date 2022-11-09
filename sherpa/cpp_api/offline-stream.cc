// sherpa/cpp_api/offline-stream.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/cpp_api/offline-stream.h"

#include <memory>
#include <string>

#include "sherpa/csrc/fbank-features.h"

namespace sherpa {

class OfflineStream::OfflineStreamImpl {
 public:
  explicit OfflineStreamImpl(kaldifeat::Fbank *fbank) : fbank_(fbank) {}

  void AcceptWaveFile(const std::string &wave_file) {
    torch::Tensor samples =
        ReadWave(wave_file, fbank_->GetFrameOptions().samp_freq).first;
    features_ = ComputeFeatures(*fbank_, {samples})[0];
  }

  void AcceptSamples(const float *samples, int32_t n) {
    torch::Tensor tensor =
        torch::from_blob(const_cast<float *>(samples), {n}, torch::kFloat);
    features_ = ComputeFeatures(*fbank_, {tensor})[0];
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
  kaldifeat::Fbank *fbank_;  // not owned
};

OfflineStream::~OfflineStream() = default;

OfflineStream::OfflineStream(kaldifeat::Fbank *fbank)
    : impl_(std::make_unique<OfflineStreamImpl>(fbank)) {}

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
