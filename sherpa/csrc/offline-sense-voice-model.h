// sherpa/csrc/offline-sense-voice-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_
#define SHERPA_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_

#include "sherpa/csrc/offline-model-config.h"
#include "sherpa/csrc/offline-sense-voice-model-meta-data.h"
#include "torch/script.h"

namespace sherpa {

class OfflineSenseVoiceModel {
 public:
  explicit OfflineSenseVoiceModel(const OfflineModelConfig &config);

  ~OfflineSenseVoiceModel();

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const;

  torch::Device Device() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_
