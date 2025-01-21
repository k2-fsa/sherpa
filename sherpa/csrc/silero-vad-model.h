// sherpa/csrc/silero-vad-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_SILERO_VAD_MODEL_H_
#define SHERPA_CSRC_SILERO_VAD_MODEL_H_

#include <memory>

#include "sherpa/csrc/vad-model-config.h"
#include "torch/script.h"

namespace sherpa {

class SileroVadModel {
 public:
  explicit SileroVadModel(const VadModelConfig &config);

  ~SileroVadModel();

  torch::Device Device() const;

  /**
   * @param samples A 2-D tensor of shape (batch_size, num_samples)
   * @returns Return A 3-D tensor of shape (batch_size, num_frames)
   */
  torch::Tensor Run(torch::Tensor samples) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_SILERO_VAD_MODEL_H_
