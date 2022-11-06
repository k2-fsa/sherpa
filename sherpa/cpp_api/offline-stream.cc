// sherpa/cpp_api/offline-stream.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/cpp_api/offline-stream.h"

#include <memory>
#include <string>

#include "sherpa/csrc/fbank_features.h"

namespace sherpa {

class OfflineStream::OfflineStreamImpl {
 public:
  OfflineStreamImpl(kaldifeat::Fbank *fbank, const std::string &wave_file) {
    torch::Tensor samples =
        ReadWave(wave_file, fbank->GetFrameOptions().samp_freq).first;
    features_ = ComputeFeatures(*fbank, {samples})[0];
  }

  OfflineStreamImpl(kaldifeat::Fbank *fbank, const float *samples, int32_t n) {
    torch::Tensor tensor =
        torch::from_blob(const_cast<float *>(samples), {n}, torch::kFloat);
    features_ = ComputeFeatures(*fbank, {tensor})[0];
  }

  OfflineStreamImpl(const float *feature, int32_t num_frames,
                    int32_t num_channels) {
    features_ = torch::from_blob(const_cast<float *>(feature),
                                 {num_frames, num_channels}, torch::kFloat)
                    .clone();
  }

  const torch::Tensor &GetFeatures() const { return features_; }

  void SetResult(const OfflineRecognitionResult &r) { result_ = r; }

  const OfflineRecognitionResult &GetResult() const { return result_; }

 private:
  torch::Tensor features_;
  OfflineRecognitionResult result_;
};

OfflineStream::~OfflineStream() = default;

OfflineStream::OfflineStream(kaldifeat::Fbank *fbank,
                             const std::string &wave_file)
    : impl_(std::make_unique<OfflineStreamImpl>(fbank, wave_file)) {}

OfflineStream::OfflineStream(kaldifeat::Fbank *fbank, const float *samples,
                             int32_t n)
    : impl_(std::make_unique<OfflineStreamImpl>(fbank, samples, n)) {}

OfflineStream::OfflineStream(const float *feature, int32_t num_frames,
                             int32_t num_channels)
    : impl_(std::make_unique<OfflineStreamImpl>(feature, num_frames,
                                                num_channels)) {}

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
