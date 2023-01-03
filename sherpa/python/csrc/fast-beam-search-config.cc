// sherpa/python/csrc/fast-beam-search.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/fast-beam-search-config.h"

#include <memory>
#include <string>

#include "sherpa/python/csrc/fast-beam-search-config.h"

namespace sherpa {

static constexpr const char *kFastBeamSearchConfigInitDoc = R"doc(
TODO
)doc";

void PybindFastBeamSearch(py::module &m) {  // NOLINT
  using PyClass = FastBeamSearchConfig;
  py::class_<PyClass>(m, "FastBeamSearchConfig")
      .def(py::init([](const std::string &lg = "", float ngram_lm_scale = 0.01,
                       float beam = 20.0, int32_t max_states = 64,
                       int32_t max_contexts = 8,
                       bool allow_partial =
                           false) -> std::unique_ptr<FastBeamSearchConfig> {
             auto config = std::make_unique<FastBeamSearchConfig>();

             config->lg = lg;
             config->ngram_lm_scale = ngram_lm_scale;
             config->beam = beam;
             config->max_states = max_states;
             config->max_contexts = max_contexts;
             config->allow_partial = allow_partial;

             return config;
           }),
           py::arg("lg") = "", py::arg("ngram_lm_scale") = 0.01,
           py::arg("beam") = 20.0, py::arg("max_states") = 64,
           py::arg("max_contexts") = 8, py::arg("allow_partial") = false,
           kFastBeamSearchConfigInitDoc)
      .def_readwrite("lg", &PyClass::lg)
      .def_readwrite("ngram_lm_scale", &PyClass::ngram_lm_scale)
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("max_states", &PyClass::max_states)
      .def_readwrite("max_contexts", &PyClass::max_contexts)
      .def_readwrite("allow_partial", &PyClass::allow_partial)
      .def("validate", &PyClass::Validate)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); });
}

}  // namespace sherpa
