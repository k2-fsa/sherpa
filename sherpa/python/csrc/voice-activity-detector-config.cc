// sherpa/python/csrc/voice-activity-detector-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/python/csrc/voice-activity-detector-config.h"

#include "sherpa/csrc/voice-activity-detector.h"
#include "sherpa/python/csrc/vad-model-config.h"

namespace sherpa {

void PybindVoiceActivityDetectorConfig(py::module *m) {
  PybindVadModelConfig(m);
  using PyClass = VoiceActivityDetectorConfig;

  py::class_<PyClass>(*m, "VoiceActivityDetectorConfig")
      .def(py::init<const VadModelConfig &, float, int32_t>(), py::arg("model"),
           py::arg("segment_size") = 10, py::arg("batch_size") = 2)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("segment_size", &PyClass::segment_size)
      .def_readwrite("batch_size", &PyClass::batch_size)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa
