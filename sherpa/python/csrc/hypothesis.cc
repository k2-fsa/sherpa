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

#include "sherpa/python/csrc/hypothesis.h"

#include <string>
#include <vector>

#include "sherpa/csrc/hypothesis.h"

namespace sherpa {

void PybindHypothesis(py::module &m) {  // NOLINT
  {
    using PyClass = Hypothesis;
    py::class_<PyClass>(m, "Hypothesis")
        .def(py::init<>())
        .def(py::init<const std::vector<int32_t> &, double>(), py::arg("ys"),
             py::arg("log_prob"))

        .def_property_readonly("key", &PyClass::Key)
        .def_property_readonly(
            "log_prob",
            [](const PyClass &self) -> double { return self.log_prob; })
        .def_property_readonly(
            "ys",
            [](const PyClass &self) -> std::vector<int32_t> { return self.ys; })
        .def_property_readonly("num_trailing_blanks",
                               [](const PyClass &self) -> int32_t {
                                 return self.num_trailing_blanks;
                               })
        .def("__str__",
             [](const PyClass &self) -> std::string { return self.ToString(); })
        .def("__repr__", [](const PyClass &self) -> std::string {
          return self.ToString();
        });
  }

  {
    using PyClass = Hypotheses;
    py::class_<PyClass>(m, "Hypotheses")
        .def(py::init<>())
        .def(py::init<std::vector<Hypothesis>>(), py::arg("hyps"))
        .def("get_most_probable", &PyClass::GetMostProbable,
             py::arg("length_norm"), py::call_guard<py::gil_scoped_release>());
  }
}

}  // namespace sherpa
