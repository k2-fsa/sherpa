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
#ifndef SHERPA_CSRC_RNNT_BEAM_SEARCH_H_
#define SHERPA_CSRC_RNNT_BEAM_SEARCH_H_

#include <vector>

#include "sherpa/csrc/hypothesis.h"
#include "sherpa/csrc/rnnt_conformer_model.h"
#include "sherpa/csrc/rnnt_emformer_model.h"
#include "sherpa/csrc/rnnt_model.h"

namespace sherpa {

/** RNN-T greedy search decoding by limiting the max symbol per frame to one.
 *
 * @param model The RNN-T model.
 *
 * @param encoder_out Output from the encoder network. Its shape is
 *                    (batch_size, T, encoder_out_dim) and its dtype is
 *                    torch::kFloat. It should be on the same device as `model`.
 *
 * @param encoder_out_lens A 1-D tensor containing the valid frames before
 *                         padding in `encoder_out`. Its dtype is torch.kLong
 *                         and its shape is (batch_size,). Also, it must be
 *                         on CPU.
 *
 * @return Return A list-of-list of token IDs containing the decoded results.
 * The returned vector has size `batch_size` and each entry contains the
 * decoded results for the corresponding input in encoder_out.
 */
std::vector<std::vector<int32_t>> GreedySearch(
    RnntModel &model,  // NOLINT
    torch::Tensor encoder_out, torch::Tensor encoder_out_length);

/** Greedy search for streaming recognition.
 *
 * @param model The stateless RNN-T Emformer model.
 * @param encoder_out A 3-D tensor of shape (N, T, C). It should be on the same
 *                    device as `model`.
 * @param decoder_out A 2-D tensor of shape (N, C). It should be on the same
 *                    device as `model`.
 * @param hyps The decoded tokens. Note: It is modified in-place.
 * @param num_trailing_blank_frames  Number of trailing blank frames. It is
 *                                   updated in-place.
 *
 * @return Return the decoder output for the next chunk.
 */
torch::Tensor StreamingGreedySearch(
    RnntModel &model,  // NOLINT
    torch::Tensor encoder_out, torch::Tensor decoder_out,
    std::vector<std::vector<int32_t>> *hyps,
    std::vector<int32_t> *num_trailing_blank_frames);

/** RNN-T modified beam search for offline recognition.
 *
 * By modified we mean that the maximum symbol per frame is limited to 1.
 *
 * @param model The RNN-T model.
 * @param encoder_out Output from the encoder network. Its shape is
 *                    (batch_size, T, encoder_out_dim) and its dtype is
 *                    torch::kFloat. It should be on the same device as `model`.
 *
 * @param encoder_out_lens A 1-D tensor containing the valid frames before
 *                         padding in `encoder_out`. Its dtype is torch.kLong
 *                         and its shape is (batch_size,). Also, it must be
 *                         on CPU.
 *
 * @param num_active_paths  Number of active paths for each utterance.
 *                          Note: Due to merging paths with identical token
 *                          sequences, the actual number of active paths for
 *                          each utterance may be smaller than this value.
 *
 * @return Return A list-of-list of token IDs containing the decoded results.
 * The returned vector has size `batch_size` and each entry contains the
 * decoded results for the corresponding input in encoder_out.
 */
std::vector<std::vector<int32_t>> ModifiedBeamSearch(
    RnntModel &model,  // NOLINT
    torch::Tensor encoder_out, torch::Tensor encoder_out_length,
    int32_t num_active_paths = 4);

/** Modified beam search for streaming recognition.
 *
 * @param model The stateless RNN-T Emformer model.
 * @param encoder_out A 3-D tensor of shape (N, T, C). It should be on the same
 *                    device as `model`.
 * @param hyps The decoded results from the previous chunk.
 * @param num_active_paths  Number of active paths for each utterance.
 *                          Note: Due to merging paths with identical token
 *                          sequences, the actual number of active paths for
 *                          each utterance may be smaller than this value.
 *
 * @return Return the decoded results for the next chunk.
 */
std::vector<Hypotheses> StreamingModifiedBeamSearch(
    RnntModel &model,  // NOLINT
    torch::Tensor encoder_out, std::vector<Hypotheses> hyps,
    int32_t num_active_paths = 4);

}  // namespace sherpa

#endif  // SHERPA_CSRC_RNNT_BEAM_SEARCH_H_
