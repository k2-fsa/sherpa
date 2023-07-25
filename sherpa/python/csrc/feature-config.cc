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
  nemo_normalize:
    Used only for NeMo CTC models. Leave it to empty if no normalization
    is used in NeMo. Current implemented method is "per_feature".
)doc";

void PybindFeatureConfig(py::module &m) {  // NOLINT
  using PyClass = FeatureConfig;
  py::class_<PyClass>(m, "FeatureConfig")
      .def(py::init([](bool normalize_samples = true,
                       const std::string &nemo_normalize =
                           "") -> std::unique_ptr<FeatureConfig> {
             auto config = std::make_unique<FeatureConfig>();

             config->normalize_samples = normalize_samples;
             config->nemo_normalize = nemo_normalize;

             return config;
           }),
           py::arg("normalize_samples") = true, py::arg("nemo_normalize") = "",
           kFeatureConfigInitDoc)
      .def_readwrite("fbank_opts", &PyClass::fbank_opts)
      .def_readwrite("normalize_samples", &PyClass::normalize_samples)
      .def_readwrite("nemo_normalize", &PyClass::nemo_normalize)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); });
}

}  // namespace sherpa
