// sherpa/csrc/offline-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_MODEL_CONFIG_H_
#define SHERPA_CSRC_OFFLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/offline-sense-voice-model-config.h"
#include "sherpa/csrc/offline-whisper-model-config.h"

namespace sherpa {

struct OfflineModelConfig {
  OfflineSenseVoiceModelConfig sense_voice;
  OfflineWhisperModelConfig whisper;

  std::string tokens;
  bool debug = false;
  bool use_gpu = false;

  OfflineModelConfig() = default;
  OfflineModelConfig(const OfflineSenseVoiceModelConfig &sense_voice,
                     const OfflineWhisperModelConfig &whisper,
                     const std::string &tokens, bool debug, bool use_gpu)
      : sense_voice(sense_voice),
        whisper(whisper),
        tokens(tokens),
        debug(debug),
        use_gpu(use_gpu) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_MODEL_CONFIG_H_
