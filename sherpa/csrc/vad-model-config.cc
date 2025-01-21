// sherpa/csrc/vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa/csrc/vad-model-config.h"

#include <sstream>
#include <string>

namespace sherpa {

void VadModelConfig::Register(ParseOptions *po) {
  silero_vad.Register(po);

  po->Register("vad-sample-rate", &sample_rate,
               "Sample rate expected by the VAD model");

  po->Register("vad-use-gpu", &use_gpu, "true to use GPU");

  po->Register("vad-debug", &debug,
               "true to display debug information when loading vad models");
}

bool VadModelConfig::Validate() const { return silero_vad.Validate(); }

std::string VadModelConfig::ToString() const {
  std::ostringstream os;

  os << "VadModelConfig(";
  os << "silero_vad=" << silero_vad.ToString() << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "use_gpu=\"" << (use_gpu ? "True" : "False") << "\", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa
