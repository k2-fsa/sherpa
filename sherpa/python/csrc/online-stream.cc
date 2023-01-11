// sherpa/python/csrc/online-stream.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/cpp_api/online-stream.h"

#include <vector>

#include "sherpa/python/csrc/online-stream.h"
#include "torch/torch.h"

namespace sherpa {

static void PybindOnlineRecognitionResult(py::module &m) {  // NOLINT
  using PyClass = OnlineRecognitionResult;
  py::class_<PyClass>(m, "OnlineRecognitionResult")
      .def_property_readonly("text",
                             [](const PyClass &self) { return self.text; })
      .def_property_readonly("tokens",
                             [](const PyClass &self) { return self.tokens; })
      .def_property_readonly(
          "timestamps", [](const PyClass &self) { return self.timestamps; })
      .def_property_readonly("segment",
                             [](const PyClass &self) { return self.segment; })
      .def_property_readonly(
          "start_time", [](const PyClass &self) { return self.start_time; })
      .def_property_readonly("is_final",
                             [](const PyClass &self) { return self.is_final; })
      .def("__str__", &PyClass::AsJsonString)
      .def("as_json_string", &PyClass::AsJsonString);
}

void PybindOnlineStream(py::module &m) {  // NOLINT
  PybindOnlineRecognitionResult(m);
  using PyClass = OnlineStream;
  py::class_<PyClass>(m, "OnlineStream")
      .def("accept_waveform", &PyClass::AcceptWaveform,
           py::arg("sampling_rate"), py::arg("waveform"),
           py::call_guard<py::gil_scoped_release>())
      .def("input_finished", &PyClass::InputFinished,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa
