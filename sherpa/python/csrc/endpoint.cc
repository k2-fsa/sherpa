// sherpa/python/csrc/endpoint.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/python/csrc/endpoint.h"

#include <memory>

#include "sherpa/cpp_api/endpoint.h"

namespace sherpa {

static constexpr const char *kEndpointRuleInitDoc = R"doc(
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

static constexpr const char *kEndpointConfigInitDoc = R"doc(
If any rule in EndpointConfig is activated, it is said that an endpointing
is detected.

Args:
  rule1:
    By default, it times out after 2.4 seconds of silence, even if
    we decoded nothing.
  rule2:
    By default, it times out after 1.2 seconds of silence after decoding
    something.
  rule3:
    By default, it times out after the utterance is 20 seconds long, regardless of
    anything else.
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
           py::arg("min_utterance_length") = 0.0f, kEndpointRuleInitDoc)
      .def("__str__", &PyClass::ToString)
      .def_readwrite("must_contain_nonsilence",
                     &PyClass::must_contain_nonsilence)
      .def_readwrite("min_trailing_silence", &PyClass::min_trailing_silence)
      .def_readwrite("min_utterance_length", &PyClass::min_utterance_length);
}

static void PybindEndpointConfig(py::module &m) {  // NOLINT
  using PyClass = EndpointConfig;
  py::class_<PyClass>(m, "EndpointConfig")
      .def(py::init([](const EndpointRule &rule1, const EndpointRule &rule2,
                       const EndpointRule &rule3) -> std::unique_ptr<PyClass> {
             auto ans = std::make_unique<PyClass>();
             ans->rule1 = rule1;
             ans->rule2 = rule2;
             ans->rule3 = rule3;
             return ans;
           }),
           py::arg("rule1") = EndpointRule(false, 2.4, 0),
           py::arg("rule2") = EndpointRule(true, 1.2, 0),
           py::arg("rule3") = EndpointRule(false, 0, 20),
           kEndpointConfigInitDoc)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def_readwrite("rule1", &PyClass::rule1)
      .def_readwrite("rule2", &PyClass::rule2)
      .def_readwrite("rule3", &PyClass::rule3);
}

void PybindEndpoint(py::module &m) {  // NOLINT
  PybindEndpointRule(m);
  PybindEndpointConfig(m);
}

}  // namespace sherpa
