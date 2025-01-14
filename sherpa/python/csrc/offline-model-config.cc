// sherpa/python/csrc/offline-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/python/csrc/offline-model-config.h"

#include <string>
#include <vector>

#include "sherpa/csrc/offline-model-config.h"
#include "sherpa/python/csrc/offline-sense-voice-model-config.h"
#include "sherpa/python/csrc/offline-whisper-model-config.h"

namespace sherpa {

void PybindOfflineModelConfig(py::module *m) {
  PybindOfflineSenseVoiceModelConfig(m);
  PybindOfflineWhisperModelConfig(m);

  using PyClass = OfflineModelConfig;
  py::class_<PyClass>(*m, "OfflineModelConfig")
      .def(py::init<const OfflineSenseVoiceModelConfig &,
                    const OfflineWhisperModelConfig &, const std::string &,
                    bool, bool>(),
           py::arg("sense_voice") = OfflineSenseVoiceModelConfig(),
           py::arg("whisper") = OfflineWhisperModelConfig(), py::arg("tokens"),
           py::arg("debug") = false, py::arg("use_gpu") = false)
      .def_readwrite("sense_voice", &PyClass::sense_voice)
      .def_readwrite("whisper", &PyClass::whisper)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("use_gpu", &PyClass::use_gpu)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa
