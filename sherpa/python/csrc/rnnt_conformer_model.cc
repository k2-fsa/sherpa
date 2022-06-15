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

#include "sherpa/python/csrc/rnnt_conformer_model.h"

#include <memory>
#include <string>

#include "sherpa/csrc/rnnt_conformer_model.h"
#include "sherpa/csrc/rnnt_model.h"
#include "torch/torch.h"

namespace sherpa {

void PybindRnntConformerModel(py::module &m) {  // NOLINT
  using PyClass = RnntConformerModel;
  py::class_<PyClass, RnntModel>(m, "RnntConformerModel")
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
      .def("encoder", &PyClass::ForwardEncoder, py::arg("features"),
           py::arg("features_length"), py::call_guard<py::gil_scoped_release>())
      .def("encoder_streaming_forward", &PyClass::StreamingForwardEncoder,
           py::arg("features"), py::arg("features_length"), py::arg("states"),
           py::arg("processed_frames"), py::arg("left_context"),
           py::arg("right_context"), py::call_guard<py::gil_scoped_release>())
      .def("get_encoder_init_states", &PyClass::GetEncoderInitStates,
           py::arg("left_context"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("subsampling_factor", &PyClass::SubSamplingFactor);
}

}  // namespace sherpa
