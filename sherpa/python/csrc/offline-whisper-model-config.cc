// sherpa/python/csrc/offline-whisper-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa/csrc/offline-whisper-model-config.h"

#include <string>
#include <vector>

#include "sherpa/python/csrc/offline-whisper-model-config.h"

namespace sherpa {

void PybindOfflineWhisperModelConfig(py::module *m) {
  using PyClass = OfflineWhisperModelConfig;
  py::class_<PyClass>(*m, "OfflineWhisperModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("model"), py::arg("language"), py::arg("task"))
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("language", &PyClass::language)
      .def_readwrite("task", &PyClass::task)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa
