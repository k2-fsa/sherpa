// sherpa/csrc/voice-activity-detector-impl.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/voice-activity-detector-impl.h"

#include <memory>

#include "sherpa/csrc/voice-activity-detector-silero-vad-impl.h"

namespace sherpa {

std::unique_ptr<VoiceActivityDetectorImpl> VoiceActivityDetectorImpl::Create(
    const VoiceActivityDetectorConfig &config) {
  return std::make_unique<VoiceActivityDetectorSileroVadImpl>(config);
}

}  // namespace sherpa
