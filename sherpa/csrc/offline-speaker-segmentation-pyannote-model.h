// sherpa/csrc/offline-speaker-segmentation-pyannote-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_H_
#define SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_H_

#include <memory>

#include "sherpa/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa/csrc/offline-speaker-segmentation-pyannote-model-meta-data.h"
#include "torch/script.h"

namespace sherpa {

class OfflineSpeakerSegmentationPyannoteModel {
 public:
  explicit OfflineSpeakerSegmentationPyannoteModel(
      const OfflineSpeakerSegmentationModelConfig &config);

  ~OfflineSpeakerSegmentationPyannoteModel();

  const OfflineSpeakerSegmentationPyannoteModelMetaData &GetModelMetaData()
      const;

  /**
   * @param x A 3-D float tensor of shape (batch_size, 1, num_samples)
   * @return Return a float tensor of
   *         shape (batch_size, num_frames, num_speakers). Note that
   *         num_speakers here uses powerset encoding.
   */
  torch::Tensor Forward(torch::Tensor x) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_SPEAKER_SEGMENTATION_PYANNOTE_MODEL_H_
