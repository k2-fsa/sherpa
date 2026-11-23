// sherpa/csrc/offline-speaker-segmentation-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/csrc/offline-speaker-segmentation-model-config.h"

#include <sstream>
#include <string>

#include "sherpa/csrc/macros.h"

namespace sherpa {

void OfflineSpeakerSegmentationModelConfig::Register(ParseOptions *po) {
  pyannote.Register(po);

  po->Register("use-gpu", &use_gpu, "true to use GPU.");

  po->Register("debug", &debug,
               "true to print model information while loading it.");
}

bool OfflineSpeakerSegmentationModelConfig::Validate() const {
  if (!pyannote.model.empty()) {
    return pyannote.Validate();
  }

  if (pyannote.model.empty()) {
    SHERPA_LOGE("You have to provide at least one speaker segmentation model");
    return false;
  }

  return true;
}

std::string OfflineSpeakerSegmentationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerSegmentationModelConfig(";
  os << "pyannote=" << pyannote.ToString() << ", ";
  os << "use_gpu=" << (use_gpu ? "True" : "False") << ", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa
