/**
 * Copyright (c)  2022  Xiaomi Corporation (authors:  Wei Kang)
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

#include "sherpa/python/csrc/ctc_model.h"

#include <memory>
#include <string>

#include "sherpa/csrc/ctc_model.h"
#include "torch/torch.h"

namespace sherpa {

void PybindCtcModel(py::module &m) {  // NOLINT
  using PyClass = CtcModel;
  py::class_<PyClass>(m, "CtcModel")
      .def_property_readonly("device",
                             [](const PyClass &self) -> py::object {
                               py::object ans =
                                   py::module_::import("torch").attr("device");
                               return ans(self.Device().str());
                             })
      .def_property_readonly("subsampling_factor", &PyClass::SubsamplingFactor);
}

}  // namespace sherpa
