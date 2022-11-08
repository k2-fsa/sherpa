/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Wei Kang)
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
#ifndef SHERPA_CSRC_CTC_BEAM_SEARCH_H_
#define SHERPA_CSRC_CTC_BEAM_SEARCH_H_

#include <vector>

#include "sherpa/csrc/fsa.h"

namespace sherpa {

/** Run CTC decode.
 * @param log_softmax_out A tensor of shape (N, T, C) containing the output
 *                        from a log_softmax layer.
 * @param log_softmax_out_lens  A tensor of shape (N,) containing the number
 *                              of valid frames in log_softmax_out before
 *                              padding.
 * @param decoding_graph  An Fsa object containing either a Ctc topology
 *                        or an HLG.
 * @param search_beam  Decoding beam, e.g. 20.  Smaller is faster, larger is
 *                     more exact (less pruning). This is the default value;
 *                     it may be modified by `min_active_states` and
 *                     `max_active_states`.
 * @param output_beam  Beam to prune output, similar to lattice-beam in Kaldi.
 *                     Relative to best path of output.
 * @param min_activate_states  Minimum number of FSA states that are allowed to
 *                             be active on any given frame for any given
 *                             intersection/composition task. This is advisory,
 *                             in that it will try not to have fewer than this
 *                             number active. Set it to zero if there is no
 *                             constraint.
 * @param max_activate_states  Maximum number of FSA states that are allowed to
 *                             be active on any given frame for any given
 *                             intersection/composition task. This is advisory,
 *                             in that it will try not to exceed that but may
 *                             not always succeed. You can use a very large
 *                             number if no constraint is needed.
 * @param subsampling_factor  The subsampling factor of the model.
 *
 * @return Return the decoding results of size `N`. ans[i] is the result
 *         for the i-th utterance. If the decoding_graph is a CtcTopo,
 *         then the decoding result contains token IDs; if the decoding_graph
 *         is an HLG, then the decoding result contains word IDs.
 *         Note: The decoding result does not contain repeats and does not
 *         contain blanks.
 */
std::vector<std::vector<int32_t>> OneBestDecoding(
    torch::Tensor log_softmax_out, torch::Tensor log_softmax_out_lens,
    Fsa decoding_graph, float search_beam = 20, float output_beam = 8,
    int32_t min_activate_states = 30, int32_t max_activate_states = 10000,
    int32_t subsampling_factor = 4);

}  // namespace sherpa

#endif  // SHERPA_CSRC_CTC_BEAM_SEARCH_H_
