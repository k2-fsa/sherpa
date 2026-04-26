// sherpa/csrc/offline-speaker-diarization-impl.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/offline-speaker-diarization-impl.h"

#include <memory>

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/offline-speaker-diarization-pyannote-impl.h"

namespace sherpa {

std::unique_ptr<OfflineSpeakerDiarizationImpl>
OfflineSpeakerDiarizationImpl::Create(
    const OfflineSpeakerDiarizationConfig &config) {
  if (!config.segmentation.pyannote.model.empty()) {
    return std::make_unique<OfflineSpeakerDiarizationPyannoteImpl>(config);
  }

  SHERPA_LOGE("Please specify a speaker segmentation model.");

  return nullptr;
}

}  // namespace sherpa
