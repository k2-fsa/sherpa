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

#include <tuple>
#include <utility>
#include <vector>

#include "sherpa/csrc/rnnt_beam_search.h"
#include "torch/torch.h"

namespace sherpa {

static constexpr const char *kGreedySearchDoc = R"doc(
RNN-T greedy search decoding by limiting the max symbol per frame to one.

Note:
  It is for offline decoding. See also :func:`streaming_greedy_search` which
  is for streaming decoding.

Args:
  model:
    The RNN-T model. It can be an instance of its subclass, such as
    :class:`RnntConformerModel`.
  encoder_out:
    Output from the encoder network. Its shape is
   ``(batch_size, T, encoder_out_dim)`` and its dtype is ``torch::kFloat``.
    It should be on the same device as ``model``.
  encoder_out_lens:
    A 1-D tensor containing the valid frames before padding in ``encoder_out``.
    Its dtype is ``torch.kLong`` and its shape is ``(batch_size,)``. Also,
    it must be on CPU.
Returns:
  Return A list-of-list of token IDs containing the decoded results. The
  returned vector has size ``batch_size`` and each entry contains the
  decoded results for the corresponding input in ``encoder_out``.
)doc";

static constexpr const char *kStreamingGreedySearchDoc = R"doc(
RNN-T greedy search for streaming recognition.

Args:
  model:
    The RNN-T model. It can be an instance of its subclass, such as
    :class:`RnntEmformerModel`.
  encoder_out:
    Output from the encoder network. Its shape is
   ``(batch_size, T, encoder_out_dim)`` and its dtype is ``torch::kFloat``.
    It should be on the same device as ``model``.
  decoder_out:
    Output from the decoder network. Its shape is
   ``(batch_size, decoder_out_dim)`` and its dtype is ``torch::kFloat``.
    It should be on the same device as ``model``.
  hyps:
    The decoded tokens from the previous chunk.
  num_trailing_blank_frames:
    Number of trailing blank frames decoded so far.

Returns:
  Return a tuple containing:
    - The decoder output for the current chunk.
    - The decoded tokens for the current chunk.
)doc";

static constexpr const char *kModifiedBeamSearchDoc = R"doc(
RNN-T modified beam search for offline recognition.

By modified we mean that the maximum symbol per frame is limited to 1.

Args:
  model:
    The RNN-T model. It can be an instance of its subclass, such as
    :class:`RnntConformerModel`.
  encoder_out:
    Output from the encoder network. Its shape is
   ``(batch_size, T, encoder_out_dim)`` and its dtype is ``torch::kFloat``.
    It should be on the same device as ``model``.
  encoder_out_lens:
    A 1-D tensor containing the valid frames before padding in ``encoder_out``.
    Its dtype is ``torch.kLong`` and its shape is ``(batch_size,)``. Also,
    it must be on CPU.
  num_active_paths
    Number of active paths for each utterance. Note: Due to merging paths with
    identical token sequences, the actual number of active path for each
    utterance may be smaller than this value.
Returns:
  Return A list-of-list of token IDs containing the decoded results. The
  returned vector has size ``batch_size`` and each entry contains the
  decoded results for the corresponding input in ``encoder_out``.
)doc";

static constexpr const char *kStreamingModifiedBeamSearchDoc = R"doc(
RNN-T modified beam search for streaming recognition.

Args:
  model:
    The RNN-T model. It can be an instance of its subclass, such as
    :class:`RnntConformerModel` and :class:`RnntConformerModel`.
  encoder_out:
    Output from the encoder network. Its shape is
   ``(batch_size, T, encoder_out_dim)`` and its dtype is ``torch::kFloat``.
    It should be on the same device as ``model``.
  hyps:
    Decoded results from the previous chunk.
  num_active_paths
    Number of active paths for each utterance. Note: Due to merging paths with
    identical token sequences, the actual number of active path for each
    utterance may be smaller than this value.

Returns:
  Decoded results for the next chunk.
)doc";

void PybindRnntBeamSearch(py::module &m) {  // NOLINT
  m.def("greedy_search", &GreedySearch, py::arg("model"),
        py::arg("encoder_out"), py::arg("encoder_out_length"),
        py::call_guard<py::gil_scoped_release>(), kGreedySearchDoc);

  m.def(
      "streaming_greedy_search",
      [](RnntModel &model, torch::Tensor encoder_out, torch::Tensor decoder_out,
         std::vector<std::vector<int32_t>> &hyps,
         std::vector<int32_t> &num_trailing_blank_frames)
          -> std::tuple<torch::Tensor, std::vector<std::vector<int32_t>>,
                        std::vector<int32_t>> {
        decoder_out = StreamingGreedySearch(model, encoder_out, decoder_out,
                                            &hyps, &num_trailing_blank_frames);
        return {decoder_out, hyps, num_trailing_blank_frames};
      },
      py::arg("model"), py::arg("encoder_out"), py::arg("decoder_out"),
      py::arg("hyps"), py::arg("num_trailing_blank_frames"),
      py::call_guard<py::gil_scoped_release>(), kStreamingGreedySearchDoc);

  m.def("modified_beam_search", &ModifiedBeamSearch, py::arg("model"),
        py::arg("encoder_out"), py::arg("encoder_out_length"),
        py::arg("num_active_paths") = 4,
        py::call_guard<py::gil_scoped_release>(), kModifiedBeamSearchDoc);

  m.def("streaming_modified_beam_search", &StreamingModifiedBeamSearch,
        py::arg("model"), py::arg("encoder_out"), py::arg("hyps"),
        py::arg("num_active_paths") = 4,
        py::call_guard<py::gil_scoped_release>(),
        kStreamingModifiedBeamSearchDoc);
}

}  // namespace sherpa
