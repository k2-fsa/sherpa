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
#ifndef SHERPA_CSRC_FSA_H_
#define SHERPA_CSRC_FSA_H_

#include "k2/torch_api.h"
#include "torch/script.h"

namespace sherpa {

struct Fsa {
  k2::FsaClassPtr fsa_ptr;

  Fsa() = default;

  /* Construct Fsa from a given graph path. The graph was saved in python by
   * `torch.save(fsa.as_dict(), filename)`.
   */
  explicit Fsa(const std::string &filename,
               torch::Device map_location = torch::kCPU);

  /* Construct Fsa from an FsaClassPtr. FsaClassPtr is exported by k2.
   */
  explicit Fsa(k2::FsaClassPtr &ptr) : fsa_ptr(ptr) {}

  /* Load an Fsa from given path. The graph was saved in python by
   * `torch.save(fsa.as_dict(), filename)`.
   */
  void Load(const std::string &filename,
            torch::Device map_location = torch::kCPU);
};

Fsa GetCtcTopo(int32_t max_token, bool modified = false,
               torch::Device map_location = torch::kCPU);

}  // namespace sherpa

#endif  // SHERPA_CSRC_FSA_H_
