// sherpa/csrc/offline-speaker-segmentation-pyannote-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/csrc/offline-speaker-segmentation-pyannote-model-config.h"

#include <sstream>
#include <string>

#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/macros.h"

namespace sherpa {

void OfflineSpeakerSegmentationPyannoteModelConfig::Register(ParseOptions *po) {
  po->Register("pyannote-model", &model,
               "Path to model.pt of the Pyannote segmentation model.");
}

bool OfflineSpeakerSegmentationPyannoteModelConfig::Validate() const {
  if (!FileExists(model)) {
    SHERPA_LOGE("Pyannote segmentation model: '%s' does not exist",
                model.c_str());
    return false;
  }

  return true;
}

std::string OfflineSpeakerSegmentationPyannoteModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerSegmentationPyannoteModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa
