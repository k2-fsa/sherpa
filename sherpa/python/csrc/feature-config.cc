// sherpa/python/csrc/feature-config.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/python/csrc/feature-config.h"

#include <memory>
#include <string>

#include "sherpa/cpp_api/feature-config.h"

namespace sherpa {

static constexpr const char *kFeatureConfigInitDoc = R"doc(
Constructor for FeatureConfig.

Args:
  fbank_opts:
    Options for computing fbank features.
  normalize_samples:
    In sherpa, the input audio samples should always be normalized to the
    range ``[-1, 1]``. If ``normalize_samples`` is ``False``, we will scale
    the input audio samples by ``32767`` inside sherpa. If ``normalize_samples``
    is ``True``, we use input audio samples as they are.
)doc";

void PybindFeatureConfig(py::module &m) {  // NOLINT
  using PyClass = FeatureConfig;
  py::class_<PyClass>(m, "FeatureConfig")
      .def(py::init([](const kaldifeat::FbankOptions &fbank_opts = {},
                       bool normalize_samples =
                           true) -> std::unique_ptr<FeatureConfig> {
             auto config = std::make_unique<FeatureConfig>();

             config->fbank_opts = fbank_opts;
             config->normalize_samples = normalize_samples;

             return config;
           }),
           py::arg("fbank_opts") = kaldifeat::FbankOptions(),
           py::arg("normalize_samples") = true, kFeatureConfigInitDoc)
      .def_readwrite("fbank_opts", &PyClass::fbank_opts)
      .def_readwrite("normalize_samples", &PyClass::normalize_samples)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); });
}

}  // namespace sherpa
