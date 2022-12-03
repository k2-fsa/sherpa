// sherpa/python/csrc/offline-recognizer.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/offline-recognizer.h"

#include "sherpa/python/csrc/offline-recognizer.h"

namespace sherpa {

static constexpr const char *kOfflineCtcDecoderConfigInitDoc = R"doc(
Constructor for Offline CTC decoder configuration.

Args:
  modified:
    ``True`` to use a modified CTC topology. ``False`` to use a standard
    CTC topology. Please visit
    `<https://k2-fsa.github.io/k2/python_api/api.html#ctc-topo>`_
    for the difference between modified and standard CTC topology.
  hlg:
    Optional. If empty, we use an ``H`` for decoding, where ``H`` is a
    CTC topology.
    If not empty, it is the path to ``HLG.pt`` and we use an ``HLG`` graph
    for decoding. Please refer to
    `<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/compile_hlg.py>`_
    for how to build an ``HLG`` graph.
  search_beam:
    Decoding beam, e.g. 20.  Smaller is faster, larger is more exact (less
    pruning). This is the default value; it may be modified by
    ``min_active_states`` and ``max_active_states``.
  output_beam:
    Beam to prune output, similar to lattice-beam in Kaldi.
    Relative to the best path of output.
  min_active_states:
    Minimum number of FSA states that are allowed to be active on any given
    frame for any given intersection/composition task. This is advisory, in
    that it will try not to have fewer than this number active. Set it to zero
    if there is no constraint.
  max_active_states:
    Maximum number of FSA states that are allowed to  be active on any given
    frame for any given intersection/composition task. This is advisory, in
    that it will try not to exceed that but may not always succeed. You can use
    a very large number if no constraint is needed.
)doc";

static void PybindOfflineCtcDecoderConfig(py::module &m) {  // NOLINT
  using PyClass = OfflineCtcDecoderConfig;
  py::class_<PyClass>(m, "OfflineCtcDecoderConfig")
      .def(py::init([](bool modified = true, const std::string &hlg = "",
                       float search_beam = 20, float output_beam = 8,
                       int32_t min_active_states = 20,
                       int32_t max_active_states =
                           10000) -> std::unique_ptr<OfflineCtcDecoderConfig> {
             auto ans = std::make_unique<OfflineCtcDecoderConfig>();

             ans->modified = modified;
             ans->hlg = hlg;
             ans->output_beam = output_beam;
             ans->search_beam = search_beam;
             ans->output_beam = output_beam;
             ans->min_active_states = min_active_states;
             ans->max_active_states = max_active_states;

             return ans;
           }),
           py::arg("modified") = true, py::arg("hlg") = "",
           py::arg("search_beam") = 20.0, py::arg("output_beam") = 8.0,
           py::arg("min_active_states") = 20,
           py::arg("max_active_states") = 10000,
           kOfflineCtcDecoderConfigInitDoc)
      .def_readwrite("modified", &PyClass::modified)
      .def_readwrite("hlg", &PyClass::hlg)
      .def_readwrite("search_beam", &PyClass::search_beam)
      .def_readwrite("output_beam", &PyClass::output_beam)
      .def_readwrite("min_active_states", &PyClass::min_active_states)
      .def_readwrite("max_active_states", &PyClass::max_active_states)
      .def("__str__", [](const PyClass &self) -> std::string {
        std::ostringstream os;
        os << "OfflineCtcDecoderConfig(";
        os << "modified=" << (self.modified ? "True" : "False") << ", ";
        os << "hlg=" << '\"' << self.hlg << '\"' << ", ";
        os << "search_beam=" << self.search_beam << ", ";
        os << "output_beam=" << self.output_beam << ", ";
        os << "min_active_states=" << self.min_active_states << ", ";
        os << "max_active_states=" << self.max_active_states << ")";
        return os.str();
      });
}

void PybindOfflineRecognizer(py::module &m) {  // NOLINT
  PybindOfflineCtcDecoderConfig(m);
}

}  // namespace sherpa
