// sherpa/csrc/offline-speaker-diarization-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_
#define SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_

#include <functional>
#include <memory>

#include "sherpa/csrc/offline-speaker-diarization.h"
namespace sherpa {

class OfflineSpeakerDiarizationImpl {
 public:
  static std::unique_ptr<OfflineSpeakerDiarizationImpl> Create(
      const OfflineSpeakerDiarizationConfig &config);

  virtual ~OfflineSpeakerDiarizationImpl() = default;

  virtual int32_t SampleRate() const = 0;

  // Note: Only config.clustering is used. All other fields in config are
  // ignored
  virtual void SetConfig(const OfflineSpeakerDiarizationConfig &config) = 0;

  virtual OfflineSpeakerDiarizationResult Process(
      torch::Tensor samples,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const = 0;
};

}  // namespace sherpa

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_
