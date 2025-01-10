// sherpa/csrc/offline-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_MODEL_CONFIG_H_
#define SHERPA_CSRC_OFFLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/offline-sense-voice-model-config.h"

namespace sherpa {

struct OfflineModelConfig {
  OfflineSenseVoiceModelConfig sense_voice;

  std::string tokens;
  bool debug = false;

  OfflineModelConfig() = default;
  OfflineModelConfig(const OfflineSenseVoiceModelConfig &sense_voice,
                     const std::string &tokens, bool debug)
      : sense_voice(sense_voice), tokens(tokens), debug(debug), {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_MODEL_CONFIG_H_
