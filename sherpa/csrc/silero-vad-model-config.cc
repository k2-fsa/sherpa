// sherpa/csrc/silero-vad-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/silero-vad-model-config.h"

#include <string>

#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/macros.h"

namespace sherpa {

void SileroVadModelConfig::Register(ParseOptions *po) {
  po->Register("silero-vad-model", &model, "Path to silero VAD model.");

  po->Register("silero-vad-threshold", &threshold,
               "Speech threshold. Silero VAD outputs speech probabilities for "
               "each audio chunk, probabilities ABOVE this value are "
               "considered as SPEECH. It is better to tune this parameter for "
               "each dataset separately, but lazy "
               "0.5 is pretty good for most datasets.");

  po->Register(
      "silero-vad-min-silence-duration", &min_silence_duration,
      "In seconds.  In the end of each speech chunk wait for "
      "--silero-vad-min-silence-duration seconds before separating it");

  po->Register("silero-vad-min-speech-duration", &min_speech_duration,
               "In seconds.  In the end of each silence chunk wait for "
               "--silero-vad-min-speech-duration seconds before separating it");
}

bool SileroVadModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_LOGE("Please provide --silero-vad-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_LOGE("Silero vad model file '%s' does not exist", model.c_str());
    return false;
  }

  if (threshold < 0.01) {
    SHERPA_LOGE(
        "Please use a larger value for --silero-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (threshold >= 1) {
    SHERPA_LOGE(
        "Please use a smaller value for --silero-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (min_silence_duration <= 0) {
    SHERPA_LOGE(
        "Please use a larger value for --silero-vad-min-silence-duration. "
        "Given: "
        "%f",
        min_silence_duration);
    return false;
  }

  if (min_speech_duration <= 0) {
    SHERPA_LOGE(
        "Please use a larger value for --silero-vad-min-speech-duration. "
        "Given: "
        "%f",
        min_speech_duration);
    return false;
  }

  return true;
}

std::string SileroVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "SileroVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ")";

  return os.str();
}

}  // namespace sherpa
