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

#include "sherpa/csrc/ctc_beam_search.h"

#include <algorithm>
#include <deque>
#include <utility>

#include "k2/torch_api.h"

namespace sherpa {

std::vector<std::vector<int32_t>> OneBestDecoding(
    torch::Tensor log_softmax_out, torch::Tensor log_softmax_out_lens,
    Fsa decoding_graph, float search_beam, float output_beam,
    int32_t min_activate_states, int32_t max_activate_states,
    int32_t subsampling_factor) {
  k2::FsaClassPtr ptr = k2::GetLattice(log_softmax_out, log_softmax_out_lens,
                                       decoding_graph.fsa_ptr, search_beam,
                                       output_beam, min_activate_states,
                                       max_activate_states, subsampling_factor);
  return k2::BestPath(ptr);
}

}  // namespace sherpa
