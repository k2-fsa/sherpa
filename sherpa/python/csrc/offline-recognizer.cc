// sherpa/python/csrc/offline-recognizer.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/offline-recognizer.h"

#include <memory>
#include <string>

#include "sherpa/python/csrc/offline-recognizer.h"

namespace sherpa {

static constexpr const char *kOfflineCtcDecoderConfigInitDoc = R"doc(
Constructor for offline CTC decoder configuration.

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

static constexpr const char *kOfflineRecognizerConfigInitDoc = R"doc(
Constructor for the offline recognizer configuration.

Args:
  nn_model:
    Path to the torchscript model. We support the following types of models:

      (1) CTC. Models from the following frameworks are supported:

        - icefall. It supports models from the ``conformer_ctc`` recipe.
        - wenet. It supports all models trained using CTC from wenet. We discard
                 the transformer decoder branch and only use the transformer
                 encoder for CTC decoding.
        - torchaudio. We support wav2vec 2.0 models from torchaudio.

      (2) Transducer. Models from the following frameworks are supported.

        - icefall. It supports models from the ``pruend_transducer_statelessX``
                   recipe.

      Please visit the following links for pre-trained CTC and transducer models:

        - `<https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc.html>`_
        - `<https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_transducer.html>`_
  tokens:
    Path to ``tokens.txt``. Note: Different frameworks use different names
    for this file. Basically, it is a text file, where each row contains two
    columns separated by space(s). The first column is a symbol and the second
    column is the corresponding integer ID of the symbol. The text file has
    as many rows as the vocabulary size of the model.
  use_gpu:
    ``False`` to use CPU for neural network computation and decoding.
    ``True`` to use GPU for neural network computation and decoding.

    .. note::

       If ``use_gpu`` is ``True``, we always use ``GPU 0``. You can use
       the environment variable ``CUDA_VISIBLE_DEVICES`` to control which
       GPU is mapped to ``GPU 0``.
  num_active_paths:
    Used only for modified_beam_search in transducer decoding. It is ignored
    if the passed ``nn_model`` is a CTC model.
  ctc_decoder_config:
    Used only when the passed ``nn_model`` is a CTC model. It is ignored if
    the passed ``nn_model`` is a transducer model.
  feat_config:
    It contains the configuration for offline fbank extractor.
  fast_beam_search_config:
    Used only for fast_beam_search in transducer decoding. It is ignored if
    the passed ``nn_model`` is a CTC model. Also, if the decoding_method is
    not ``fast_beam_search``, it is ignored.
  decoding_method:
    Used only when the passed ``nn_model`` is a transducer model.
    Valid values are: ``greedy_search``, ``modified_beam_search``, and
    ``fast_beam_search``.
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
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def("validate", &PyClass::Validate);
}

static void PybindOfflineRecognizerConfig(py::module &m) {  // NOLINT
  using PyClass = OfflineRecognizerConfig;
  py::class_<PyClass>(m, "OfflineRecognizerConfig")
      .def(py::init([](const std::string &nn_model, const std::string &tokens,
                       bool use_gpu = false, int32_t num_active_paths = 4,
                       const OfflineCtcDecoderConfig &ctc_decoder_config = {},
                       const FeatureConfig &feat_config = {},
                       const FastBeamSearchConfig &fast_beam_search_config = {},
                       const std::string &decoding_method = "greedy_search")
                        -> std::unique_ptr<OfflineRecognizerConfig> {
             auto config = std::make_unique<OfflineRecognizerConfig>();

             config->ctc_decoder_config = ctc_decoder_config;
             config->feat_config = feat_config;
             config->fast_beam_search_config = fast_beam_search_config;
             config->nn_model = nn_model;
             config->tokens = tokens;
             config->use_gpu = use_gpu;
             config->decoding_method = decoding_method;
             config->num_active_paths = num_active_paths;

             return config;
           }),
           py::arg("nn_model"), py::arg("tokens"), py::arg("use_gpu") = false,
           py::arg("num_active_paths") = 4,
           py::arg("ctc_decoder_config") = OfflineCtcDecoderConfig(),
           py::arg("feat_config") = FeatureConfig(),
           py::arg("fast_beam_search_config") = FastBeamSearchConfig(),
           py::arg("decoding_method") = "greedy_search",
           kOfflineRecognizerConfigInitDoc)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def_readwrite("ctc_decoder_config", &PyClass::ctc_decoder_config)
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("fast_beam_search_config",
                     &PyClass::fast_beam_search_config)
      .def_readwrite("nn_model", &PyClass::nn_model)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("use_gpu", &PyClass::use_gpu)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("num_active_paths", &PyClass::num_active_paths)
      .def("validate", &PyClass::Validate);
}

void PybindOfflineRecognizer(py::module &m) {  // NOLINT
  PybindOfflineCtcDecoderConfig(m);
  PybindOfflineRecognizerConfig(m);
}

}  // namespace sherpa
