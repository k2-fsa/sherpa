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

#include "sherpa/python/csrc/rnnt_model.h"

#include <memory>

#include "sherpa/csrc/rnnt_model.h"
#include "torch/torch.h"

namespace sherpa {

void PybindRnntModel(py::module &m) {
  using PyClass = RnntModel;
  py::class_<PyClass>(m, "RnntModel")
      .def(py::init([](const std::string &filename,
                       py::object device = py::str("cpu"),
                       bool optimize_for_inference =
                           false) -> std::unique_ptr<RnntModel> {
             std::string device_str =
                 device.is_none() ? "cpu" : py::str(device);
             return std::make_unique<RnntModel>(
                 filename, torch::Device(device_str), optimize_for_inference);
           }),
           py::arg("filename"), py::arg("device") = py::str("cpu"),
           py::arg("optimize_for_inference") = false)
      .def("encoder", &PyClass::ForwardEncoder, py::arg("features"),
           py::arg("features_length"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("device", [](const PyClass &self) -> py::object {
        py::object ans = py::module_::import("torch").attr("device");
        return ans(self.Device().str());
      });
}

}  // namespace sherpa
