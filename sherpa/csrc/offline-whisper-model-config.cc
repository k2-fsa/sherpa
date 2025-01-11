// sherpa/csrc/offline-whisper-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/offline-whisper-model-config.h"

#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/macros.h"

namespace sherpa {

void OfflineWhisperModelConfig::Register(ParseOptions *po) {
  po->Register("whisper-model", &model,
               "Path to the torchscript model of whisper");

  po->Register(
      "whisper-language", &language,
      "The spoken language in the input audio file. Example values: "
      "en, de, fr, zh, jp. If it is not given for a multilingual model, we will"
      " infer the language from the input audio file. "
      "Please refer to "
      "https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10"
      " for valid values. Note that for non-multilingual models, it supports "
      "only 'en'");

  po->Register("whisper-task", &task,
               "Valid values: transcribe, translate. "
               "Note that for non-multilingual models, it supports "
               "only 'transcribe'");
}

bool OfflineWhisperModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_LOGE("Please provide --whisper-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_LOGE("whisper model file '%s' does not exist", model.c_str());
    return false;
  }

  if (task != "translate" && task != "transcribe") {
    SHERPA_LOGE(
        "--whisper-task supports only translate and transcribe. Given: %s",
        task.c_str());

    return false;
  }

  return true;
}

std::string OfflineWhisperModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineWhisperModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "language=\"" << language << "\", ";
  os << "task=\"" << task << "\")";

  return os.str();
}

}  // namespace sherpa
