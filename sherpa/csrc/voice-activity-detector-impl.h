// sherpa/csrc/voice-activity-detector-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_IMPL_H_
#define SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_IMPL_H_
#include <memory>
#include <vector>

#include "sherpa/csrc/voice-activity-detector.h"
#include "torch/script.h"

namespace sherpa {

class VoiceActivityDetectorImpl {
 public:
  static std::unique_ptr<VoiceActivityDetectorImpl> Create(
      const VoiceActivityDetectorConfig &config);

  virtual ~VoiceActivityDetectorImpl() = default;

  virtual const VoiceActivityDetectorConfig &GetConfig() const = 0;

  virtual std::vector<SpeechSegment> Process(torch::Tensor samples) = 0;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_IMPL_H_
