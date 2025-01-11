// sherpa/csrc/offline-whisper-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_
#define SHERPA_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_

#include <string>

#include "sherpa/cpp_api/parse-options.h"

namespace sherpa {

struct OfflineWhisperModelConfig {
  std::string model;

  // Available languages can be found at
  // https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
  //
  // Note: For non-multilingual models, it supports only "en"
  //
  // If empty, we will infer it from the input audio file when
  // the model is multilingual.
  std::string language;

  // Valid values are transcribe and translate
  //
  // Note: For non-multilingual models, it supports only "transcribe"
  std::string task = "transcribe";

  OfflineWhisperModelConfig() = default;
  OfflineWhisperModelConfig(const std::string &model,
                            const std::string &language,
                            const std::string &task)
      : model(model), language(language), task(task) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_
