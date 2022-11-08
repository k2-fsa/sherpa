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

#include "sherpa/python/csrc/fsa.h"

#include <memory>
#include <string>

#include "sherpa/csrc/fsa.h"
#include "torch/torch.h"

namespace sherpa {

void PybindFsa(py::module &m) {  // NOLINT
  using PyClass = Fsa;
  py::class_<PyClass>(m, "Fsa")
      .def(py::init([](const std::string &filename,
                       py::object map_location =
                           py::str("cpu")) -> std::unique_ptr<PyClass> {
             std::string device_str =
                 map_location.is_none() ? "cpu" : py::str(map_location);
             return std::make_unique<PyClass>(filename,
                                              torch::Device(device_str));
           }),
           py::arg("filename"), py::arg("map_location") = py::str("cpu"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "load",
          [](PyClass &self, const std::string &filename,
             py::object map_location = py::str("cpu")) -> void {
            std::string device_str =
                map_location.is_none() ? "cpu" : py::str(map_location);
            self.Load(filename, torch::Device(device_str));
            return;
          },
          py::arg("filename"), py::arg("map_location") = py::str("cpu"),
          py::call_guard<py::gil_scoped_release>());
  m.def(
      "get_ctc_topo",
      [](int32_t max_token, bool modified = false,
         py::object device = py::str("cpu")) -> Fsa {
        std::string device_str = device.is_none() ? "cpu" : py::str(device);
        return GetCtcTopo(max_token, modified, torch::Device(device_str));
      },
      py::arg("max_token"), py::arg("modified") = false,
      py::arg("device") = py::str("cpu"),
      py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa
