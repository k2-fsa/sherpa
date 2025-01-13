// sherpa/csrc/offline-whisper-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_WHISPER_MODEL_H_
#define SHERPA_CSRC_OFFLINE_WHISPER_MODEL_H_

#include <memory>

#include "sherpa/csrc/offline-model-config.h"
#include "torch/script.h"

namespace sherpa {

class OfflineWhisperModel {
 public:
  explicit OfflineWhisperModel(const OfflineModelConfig &config);

  ~OfflineWhisperModel();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_WHISPER_MODEL_H_
