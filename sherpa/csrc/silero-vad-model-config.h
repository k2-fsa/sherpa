// sherpa/csrc/silero-vad-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_SILERO_VAD_MODEL_CONFIG_H_
#define SHERPA_CSRC_SILERO_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa/cpp_api/parse-options.h"

namespace sherpa {

struct SileroVadModelConfig {
  std::string model;

  // threshold to classify a segment as speech
  //
  // If the predicted probability of a segment is larger than this
  // value, then it is classified as speech.
  float threshold = 0.5;

  float min_silence_duration = 0.5;  // in seconds

  float min_speech_duration = 0.25;  // in seconds

  SileroVadModelConfig() = default;
  SileroVadModelConfig(const std::string &model, float threshold,
                       float min_silence_duration, float min_speech_duration)
      : model(model),
        threshold(threshold),
        min_silence_duration(min_silence_duration),
        min_speech_duration(min_speech_duration) {}

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_SILERO_VAD_MODEL_CONFIG_H_
