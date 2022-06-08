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
#include "sherpa/python/csrc/rnnt_beam_search.h"

#include <utility>
#include <vector>

#include "sherpa/csrc/rnnt_beam_search.h"
#include "torch/torch.h"

namespace sherpa {

void PybindRnntBeamSearch(py::module &m) {  // NOLINT
  m.def("greedy_search", &GreedySearch, py::arg("model"),
        py::arg("encoder_out"), py::arg("encoder_out_length"),
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "streaming_greedy_search",
      [](RnntConformerModel &model, torch::Tensor encoder_out,
         torch::Tensor decoder_out, std::vector<std::vector<int32_t>> &hyps)
          -> std::pair<torch::Tensor, std::vector<std::vector<int32_t>>> {
        decoder_out =
            StreamingGreedySearch(model, encoder_out, decoder_out, &hyps);
        return {decoder_out, hyps};
      },
      py::arg("model"), py::arg("encoder_out"), py::arg("decoder_out"),
      py::arg("hyps"), py::call_guard<py::gil_scoped_release>());

  m.def(
      "streaming_greedy_search",
      [](RnntEmformerModel &model, torch::Tensor encoder_out,
         torch::Tensor decoder_out, std::vector<std::vector<int32_t>> &hyps)
          -> std::pair<torch::Tensor, std::vector<std::vector<int32_t>>> {
        decoder_out =
            StreamingGreedySearch(model, encoder_out, decoder_out, &hyps);
        return {decoder_out, hyps};
      },
      py::arg("model"), py::arg("encoder_out"), py::arg("decoder_out"),
      py::arg("hyps"), py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa
