// sherpa/csrc/vad-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_VAD_MODEL_CONFIG_H_
#define SHERPA_CSRC_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/silero-vad-model-config.h"

namespace sherpa {

struct VadModelConfig {
  SileroVadModelConfig silero_vad;

  int32_t sample_rate = 16000;
  bool use_gpu = false;

  // true to show debug information when loading models
  bool debug = false;

  VadModelConfig() = default;

  VadModelConfig(const SileroVadModelConfig &silero_vad, int32_t sample_rate,
                 bool use_gpu, bool debug)
      : silero_vad(silero_vad),
        sample_rate(sample_rate),
        use_gpu(use_gpu),
        debug(debug) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_VAD_MODEL_CONFIG_H_
