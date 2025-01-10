// sherpa/csrc/offline-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/csrc/offline-model-config.h"

#include <string>

#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/macros.h"

namespace sherpa {

void OfflineModelConfig::Register(ParseOptions *po) {
  sense_voice.Register(po);

  // TODO(fangjun): enable it
  // po->Register("tokens", &tokens, "Path to tokens.txt");

  po->Register("debug", &debug,
               "true to print model information while loading it.");
}

bool OfflineModelConfig::Validate() const {
  if (!FileExists(tokens)) {
    SHERPA_LOGE("tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (!sense_voice.model.empty()) {
    return sense_voice.Validate();
  }

  return true;
}

std::string OfflineModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineModelConfig(";
  os << "sense_voice=" << sense_voice.ToString() << ", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa
