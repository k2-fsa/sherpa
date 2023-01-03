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

#include "sherpa/python/csrc/rnnt_model.h"

#include <memory>
#include <string>

#include "sherpa/csrc/rnnt_model.h"
#include "torch/torch.h"

namespace sherpa {

void PybindRnntModel(py::module &m) {  // NOLINT
  using PyClass = RnntModel;
  py::class_<PyClass>(m, "RnntModel")
      .def("decoder_forward", &PyClass::ForwardDecoder,
           py::arg("decoder_input"), py::call_guard<py::gil_scoped_release>())
      .def("joiner_forward", &PyClass::ForwardJoiner,
           py::arg("projected_encoder_out"), py::arg("projected_decoder_out"),
           py::call_guard<py::gil_scoped_release>())
      .def("forward_decoder_proj", &PyClass::ForwardDecoderProj,
           py::arg("decoder_out"), py::call_guard<py::gil_scoped_release>())
      .def("forward_encoder_proj", &PyClass::ForwardEncoderProj,
           py::arg("encoder_out"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("device",
                             [](const PyClass &self) -> py::object {
                               py::object ans =
                                   py::module_::import("torch").attr("device");
                               return ans(self.Device().str());
                             })
      .def_property_readonly("blank_id", &PyClass::BlankId)
      .def_property_readonly("unk_id", &PyClass::UnkId)
      .def_property_readonly("vocab_size", &PyClass::VocabSize)
      .def_property_readonly("context_size", &PyClass::ContextSize)
      .def_property_readonly("subsampling_factor", &PyClass::SubsamplingFactor);
}

}  // namespace sherpa
