// sherpa/python/csrc/speaker-embedding-extractor.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/python/csrc/speaker-embedding-extractor.h"

#include <string>
#include <vector>

#include "sherpa/csrc/speaker-embedding-extractor.h"

namespace sherpa {

static void PybindSpeakerEmbeddingExtractorConfig(py::module *m) {
  using PyClass = SpeakerEmbeddingExtractorConfig;
  py::class_<PyClass>(*m, "SpeakerEmbeddingExtractorConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, bool, bool>(), py::arg("model"),
           py::arg("use_gpu") = false, py::arg("debug") = false)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("use_gpu", &PyClass::use_gpu)
      .def_readwrite("debug", &PyClass::debug)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindSpeakerEmbeddingExtractor(py::module *m) {
  PybindSpeakerEmbeddingExtractorConfig(m);

  using PyClass = SpeakerEmbeddingExtractor;
  py::class_<PyClass>(*m, "SpeakerEmbeddingExtractor")
      .def(py::init<const SpeakerEmbeddingExtractorConfig &>(),
           py::arg("config"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("dim", &PyClass::Dim)
      .def("create_stream", &PyClass::CreateStream,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "compute",
          [](PyClass &self, OfflineStream *s) { return self.Compute(s); },
          py::arg("s"), py::call_guard<py::gil_scoped_release>())
      .def(
          "compute",
          [](PyClass &self, std::vector<OfflineStream *> &ss) {
            return self.Compute(ss.data(), ss.size());
          },
          py::arg("ss"), py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa
