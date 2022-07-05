/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang,
 *                                                   Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sherpa/python/csrc/rnnt_conv_emformer_model.h"

#include <memory>
#include <string>

#include "sherpa/csrc/rnnt_conv_emformer_model.h"
#include "sherpa/csrc/rnnt_model.h"
#include "torch/torch.h"

namespace sherpa {

void PybindRnntConvEmformerModel(py::module &m) {  // NOLINT
  using PyClass = RnntConvEmformerModel;
  py::class_<PyClass, RnntModel>(m, "RnntConvEmformerModel")
      .def(py::init([](const std::string &filename,
                       py::object device = py::str("cpu"),
                       bool optimize_for_inference =
                           false) -> std::unique_ptr<PyClass> {
             std::string device_str =
                 device.is_none() ? "cpu" : py::str(device);
             return std::make_unique<PyClass>(
                 filename, torch::Device(device_str), optimize_for_inference);
           }),
           py::arg("filename"), py::arg("device") = py::str("cpu"),
           py::arg("optimize_for_inference") = false)
      .def("encoder_streaming_forward", &PyClass::StreamingForwardEncoder,
           py::arg("features"), py::arg("features_length"),
           py::arg("num_processed_frames"), py::arg("states"),
           py::call_guard<py::gil_scoped_release>())
      .def("get_encoder_init_states", &PyClass::GetEncoderInitStates,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("chunk_length", &PyClass::ChunkLength)
      .def_property_readonly("right_context_length",
                             &PyClass::RightContextLength)
      .def_property_readonly("pad_length", &PyClass::PadLength);
}

}  // namespace sherpa
