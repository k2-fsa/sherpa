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

Fsa GetLattice(torch::Tensor log_softmax_out,
               torch::Tensor log_softmax_out_lens, Fsa decoding_graph,
               float search_beam = 20, float output_beam = 8,
               int32_t min_activate_states = 30,
               int32_t max_activate_states = 10000,
               int32_t subsampling_factor = 4);

std::vector<std::vector<int32_t>> BestPath(const Fsa &lattice);

}  // namespace sherpa

#endif  // SHERPA_CSRC_CTC_BEAM_SEARCH_H_
