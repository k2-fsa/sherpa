// sherpa/csrc/offline-speaker-segmentation-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_MODEL_CONFIG_H_
#define SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_MODEL_CONFIG_H_

#include <string>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/offline-speaker-segmentation-pyannote-model-config.h"

namespace sherpa {

struct OfflineSpeakerSegmentationModelConfig {
  OfflineSpeakerSegmentationPyannoteModelConfig pyannote;

  bool use_gpu = false;
  bool debug = false;

  OfflineSpeakerSegmentationModelConfig() = default;

  explicit OfflineSpeakerSegmentationModelConfig(
      const OfflineSpeakerSegmentationPyannoteModelConfig &pyannote,
      bool use_gpu, bool debug)
      : pyannote(pyannote), use_gpu(use_gpu), debug(debug) {}

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_MODEL_CONFIG_H_
