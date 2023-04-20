// sherpa/python/csrc/offline-stream.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/offline-stream.h"

#include <vector>

#include "sherpa/python/csrc/offline-stream.h"
#include "torch/torch.h"

namespace sherpa {

static constexpr const char *kOfflineStreamAcceptSamplesVectorDoc = R"doc(
Accept samples from a list of floats.

Args:
  samples:
    It contains audio samples normalized to the range ``[-1, 1].``
    Note: The sampling rate of the samples should match the one expected
    by the feature extractor.
)doc";

static constexpr const char *kOfflineStreamAcceptSamplesTensorDoc = R"doc(
Accept samples from a 1-D float32 tensor .

Args:
  samples:
    It contains audio samples normalized to the range ``[-1, 1].``
    Note: The sampling rate of the samples should match the one expected
    by the feature extractor.
)doc";

static void PybindOfflineRecognitionResult(py::module &m) {  // NOLINT
  using PyClass = OfflineRecognitionResult;
  py::class_<PyClass>(m, "OfflineRecognitionResult")
      .def_property_readonly("text",
                             [](const PyClass &self) { return self.text; })
      .def_property_readonly("tokens",
                             [](const PyClass &self) { return self.tokens; })
      .def_property_readonly(
          "timestamps", [](const PyClass &self) { return self.timestamps; })
      .def("__str__", &PyClass::AsJsonString)
      .def("as_json_string", &PyClass::AsJsonString);
}

void PybindOfflineStream(py::module &m) {  // NOLINT
  PybindOfflineRecognitionResult(m);
  using PyClass = OfflineStream;

  py::class_<PyClass> stream(m, "OfflineStream");
  stream
      .def("accept_wave_file", &PyClass::AcceptWaveFile,
           py::call_guard<py::gil_scoped_release>(), py::arg("filename"))
      .def(
          "accept_samples",
          [](PyClass &self, const std::vector<float> &samples) {
            self.AcceptSamples(samples.data(), samples.size());
          },
          py::arg("samples"), py::call_guard<py::gil_scoped_release>(),
          kOfflineStreamAcceptSamplesVectorDoc)
      .def(
          "accept_samples",
          [](PyClass &self, torch::Tensor samples) {
            samples = samples.contiguous().cpu();
            self.AcceptSamples(samples.data_ptr<float>(), samples.numel());
          },
          py::arg("samples"), py::call_guard<py::gil_scoped_release>(),
          kOfflineStreamAcceptSamplesTensorDoc)
      .def_property_readonly("result", &PyClass::GetResult);

  // alias
  stream.attr("accept_waveform") = stream.attr("accept_samples");
}

}  // namespace sherpa
