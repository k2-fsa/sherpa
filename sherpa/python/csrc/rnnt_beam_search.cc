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
#include "sherpa/csrc/rnnt_beam_search.h"

#include "sherpa/python/csrc/rnnt_beam_search.h"
#include "torch/torch.h"

namespace sherpa {

void PybindRnntBeamSearch(py::module &m) {
  m.def("greedy_search", &GreedySearch, py::arg("model"),
        py::arg("encoder_out"), py::arg("encoder_out_length"),
        py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa
