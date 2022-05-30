/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "sherpa/python/csrc/rnnt_emformer_model.h"

#include <memory>
#include <string>

#include "sherpa/csrc/rnnt_emformer_model.h"
#include "torch/torch.h"

namespace sherpa {

void PybindRnntEmformerModel(py::module &m) {  // NOLINT
  using PyClass = RnntEmformerModel;
  py::class_<PyClass>(m, "RnntEmformerModel")
      .def(py::init([](const std::string &filename,
                       py::object device = py::str("cpu"),
                       bool optimize_for_inference =
                           false) -> std::unique_ptr<RnntEmformerModel> {
             std::string device_str =
                 device.is_none() ? "cpu" : py::str(device);
             return std::make_unique<RnntEmformerModel>(
                 filename, torch::Device(device_str), optimize_for_inference);
           }),
           py::arg("filename"), py::arg("device") = py::str("cpu"),
           py::arg("optimize_for_inference") = false)
      .def("encoder_streaming_forward", &PyClass::StreamingForwardEncoder,
           py::arg("features"), py::arg("features_length"),
           py::arg("states") = py::none(),
           py::call_guard<py::gil_scoped_release>())
      .def("decoder_forward", &PyClass::ForwardDecoder,
           py::arg("decoder_input"), py::call_guard<py::gil_scoped_release>())
      .def("get_encoder_init_states", &PyClass::GetEncoderInitStates,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("device",
                             [](const PyClass &self) -> py::object {
                               py::object ans =
                                   py::module_::import("torch").attr("device");
                               return ans(self.Device().str());
                             })
      .def_property_readonly("blank_id", &PyClass::BlankId)
      .def_property_readonly("context_size", &PyClass::ContextSize)
      .def_property_readonly("segment_length", &PyClass::SegmentLength)
      .def_property_readonly("right_context_length",
                             &PyClass::RightContextLength);
}

}  // namespace sherpa
