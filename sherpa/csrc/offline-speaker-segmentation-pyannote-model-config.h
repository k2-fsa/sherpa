// sherpa/csrc/offline-speaker-segmentation-pyannote-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_CONFIG_H_
#define SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_CONFIG_H_
#include <string>

#include "sherpa/cpp_api/parse-options.h"

namespace sherpa {

struct OfflineSpeakerSegmentationPyannoteModelConfig {
  std::string model;

  OfflineSpeakerSegmentationPyannoteModelConfig() = default;

  explicit OfflineSpeakerSegmentationPyannoteModelConfig(
      const std::string &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_CONFIG_H_
