// sherpa/python/csrc/silero-vad-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/python/csrc/silero-vad-model-config.h"

#include <string>

#include "sherpa/csrc/silero-vad-model-config.h"

namespace sherpa {

void PybindSileroVadModelConfig(py::module *m) {
  using PyClass = SileroVadModelConfig;
  py::class_<PyClass>(*m, "SileroVadModelConfig")
      .def(py::init<const std::string &, float, float, float>(),
           py::arg("model") = "", py::arg("threshold") = 0.5,
           py::arg("min_silence_duration") = 0.5,
           py::arg("min_speech_duration") = 0.25)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("threshold", &PyClass::threshold)
      .def_readwrite("min_silence_duration", &PyClass::min_silence_duration)
      .def_readwrite("min_speech_duration", &PyClass::min_speech_duration)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa
