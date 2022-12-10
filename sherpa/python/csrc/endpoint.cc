// sherpa/python/csrc/endpoint.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/python/csrc/endpoint.h"

#include <memory>

#include "sherpa/cpp_api/endpoint.h"

namespace sherpa {

static constexpr const char *kEndpointInitDoc = R"doc(
Constructor for EndpointRule.

Args:
  must_contain_nonsilence:
    If True, for this endpointing rule to apply there must be nonsilence in the
    best-path traceback. For decoding, a non-blank token is considered as
    non-silence.
  min_trailing_silence:
    This endpointing rule requires duration of trailing silence (in seconds) to
    be >= this value.
  min_utterance_length:
    This endpointing rule requires utterance-length (in seconds) to be >= this
    value.
)doc";

static void PybindEndpointRule(py::module &m) {  // NOLINT
  using PyClass = EndpointRule;
  py::class_<PyClass>(m, "EndpointRule")
      .def(py::init([](bool must_contain_nonsilence = true,
                       float min_trailing_silence = 2.0,
                       float min_utterance_length =
                           0.0f) -> std::unique_ptr<PyClass> {
             auto ans = std::make_unique<PyClass>();

             ans->must_contain_nonsilence = must_contain_nonsilence;
             ans->min_trailing_silence = min_trailing_silence;
             ans->min_utterance_length = min_utterance_length;

             return ans;
           }),
           py::arg("must_contain_nonsilence") = true,
           py::arg("min_trailing_silence") = 2.0,
           py::arg("min_utterance_length") = 0.0f, kEndpointInitDoc)
      .def("__str__", &PyClass::ToString)
      .def_readwrite("must_contain_nonsilence",
                     &PyClass::must_contain_nonsilence)
      .def_readwrite("min_trailing_silence", &PyClass::min_trailing_silence)
      .def_readwrite("min_utterance_length", &PyClass::min_utterance_length);
}

void PybindEndpoint(py::module &m) {  // NOLINT
  PybindEndpointRule(m);
}

}  // namespace sherpa
