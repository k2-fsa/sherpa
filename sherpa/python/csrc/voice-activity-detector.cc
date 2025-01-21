// sherpa/python/csrc/voice-activity-detector.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/python/csrc/voice-activity-detector.h"

#include <iomanip>

#include "sherpa/csrc/voice-activity-detector.h"
#include "sherpa/python/csrc/voice-activity-detector-config.h"

namespace sherpa {

void PybindSpeechSegment(py::module *m) {
  using PyClass = SpeechSegment;
  py::class_<PyClass>(*m, "SpeechSegment")
      .def_property_readonly("start",
                             [](const PyClass &self) { return self.start; })
      .def_property_readonly("end",
                             [](const PyClass &self) { return self.end; })
      .def("__str__", [](const PyClass &self) {
        std::ostringstream os;
        os << "SpeechSegment(";
        os << std::setprecision(3) << self.start << ", ";
        os << std::setprecision(3) << self.end << ")";
        return os.str();
      });
}

void PybindVoiceActivityDetector(py::module *m) {
  PybindVoiceActivityDetectorConfig(m);
  PybindSpeechSegment(m);

  using PyClass = VoiceActivityDetector;
  py::class_<PyClass>(*m, "VoiceActivityDetector")
      .def(py::init<const VoiceActivityDetectorConfig &>(), py::arg("config"))
      .def_property_readonly("config", &PyClass::GetConfig)
      .def("process", &PyClass::Process, py::arg("samples"),
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa
