// sherpa/python/csrc/online-recognizer.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/cpp_api/online-recognizer.h"

#include <memory>
#include <string>
#include <vector>

#include "sherpa/python/csrc/online-recognizer.h"

namespace sherpa {

static void PybindOnlineRecognizerConfig(py::module &m) {  // NOLINT
  using PyClass = OnlineRecognizerConfig;
  py::class_<PyClass>(m, "OnlineRecognizerConfig")
      .def(py::init([](const std::string &nn_model, const std::string &tokens,
                       const std::string &encoder_model = {},
                       const std::string &decoder_model = {},
                       const std::string &joiner_model = {},
                       bool use_gpu = false, bool use_endpoint = false,
                       const std::string &decoding_method = "greedy_search",
                       int32_t num_active_paths = 4, int32_t left_context = 64,
                       int32_t right_context = 0, int32_t chunk_size = 16,
                       const FeatureConfig &feat_config = {},
                       const EndpointConfig &endpoint_config = {},
                       const FastBeamSearchConfig &fast_beam_search_config = {})
                        -> std::unique_ptr<OnlineRecognizerConfig> {
             auto ans = std::make_unique<OnlineRecognizerConfig>();

             ans->feat_config = feat_config;
             ans->endpoint_config = endpoint_config;
             ans->fast_beam_search_config = fast_beam_search_config;
             ans->nn_model = nn_model;
             ans->tokens = tokens;
             ans->encoder_model = encoder_model;
             ans->decoder_model = decoder_model;
             ans->joiner_model = joiner_model;
             ans->use_gpu = use_gpu;
             ans->use_endpoint = use_endpoint;
             ans->decoding_method = decoding_method;
             ans->num_active_paths = num_active_paths;
             ans->left_context = left_context;
             ans->right_context = right_context;
             ans->chunk_size = chunk_size;

             return ans;
           }),
           py::arg("nn_model"), py::arg("tokens"),
           py::arg("encoder_model") = "", py::arg("decoder_model") = "",
           py::arg("joiner_model") = "", py::arg("use_gpu") = false,
           py::arg("use_endpoint") = false,
           py::arg("decoding_method") = "greedy_search",
           py::arg("num_active_paths") = 4, py::arg("left_context") = 64,
           py::arg("right_context") = 0, py::arg("chunk_size") = 16,
           py::arg("feat_config") = FeatureConfig(),
           py::arg("endpoint_config") = EndpointConfig(),
           py::arg("fast_beam_search_config") = FastBeamSearchConfig())
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("endpoint_config", &PyClass::endpoint_config)
      .def_readwrite("fast_beam_search_config",
                     &PyClass::fast_beam_search_config)
      .def_readwrite("nn_model", &PyClass::nn_model)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("encoder_model", &PyClass::encoder_model)
      .def_readwrite("decoder_model", &PyClass::decoder_model)
      .def_readwrite("joiner_model", &PyClass::joiner_model)
      .def_readwrite("use_gpu", &PyClass::use_gpu)
      .def_readwrite("use_endpoint", &PyClass::use_endpoint)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("num_active_paths", &PyClass::num_active_paths)
      .def_readwrite("left_context", &PyClass::left_context)
      .def_readwrite("right_context", &PyClass::right_context)
      .def_readwrite("chunk_size", &PyClass::chunk_size)
      .def("validate", &PyClass::Validate)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); });
}

void PybindOnlineRecognizer(py::module &m) {  // NOLINT
  PybindOnlineRecognizerConfig(m);
  using PyClass = OnlineRecognizer;
  py::class_<PyClass>(m, "OnlineRecognizer")
      .def(py::init<const OnlineRecognizerConfig &>(), py::arg("config"))
      .def("create_stream", &PyClass::CreateStream,
           py::call_guard<py::gil_scoped_release>())
      .def("is_ready", &PyClass::IsReady, py::arg("s"),
           py::call_guard<py::gil_scoped_release>())
      .def("is_endpoint", &PyClass::IsEndpoint, py::arg("s"),
           py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream, py::arg("s"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](PyClass &self, std::vector<OnlineStream *> &ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::arg("ss"), py::call_guard<py::gil_scoped_release>())
      .def("get_result", &PyClass::GetResult, py::arg("s"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("config", &PyClass::GetConfig,
                             py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa
