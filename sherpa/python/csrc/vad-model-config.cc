// sherpa/python/csrc/vad-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/python/csrc/vad-model-config.h"

#include "sherpa/csrc/vad-model-config.h"
#include "sherpa/python/csrc/silero-vad-model-config.h"

namespace sherpa {

void PybindVadModelConfig(py::module *m) {
  PybindSileroVadModelConfig(m);
  using PyClass = VadModelConfig;

  py::class_<PyClass>(*m, "VadModelConfig")
      .def(py::init<const SileroVadModelConfig &, int32_t, bool, bool>(),
           py::arg("silero_vad"), py::arg("sample_rate") = 16000,
           py::arg("use_gpu") = false, py::arg("debug") = false)
      .def_readwrite("silero_vad", &PyClass::silero_vad)
      .def_readwrite("sample_rate", &PyClass::sample_rate)
      .def_readwrite("use_gpu", &PyClass::use_gpu)
      .def_readwrite("debug", &PyClass::debug)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa
