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

#include "sherpa/csrc/fsa.h"

namespace sherpa {

Fsa::Fsa(std::string &filename, torch::Device map_location) {
  fsa_ptr = k2::LoadFsaClass(filename, map_location);
}

void Fsa::Load(std::string &filename, torch::Device map_location) {
  fsa_ptr = k2::LoadFsaClass(filename, map_location);
}

Fsa GetCtcTopo(int32_t max_token, bool modified, torch::Device map_location) {
  auto fsa_ptr = k2::GetCtcTopo(max_token, modified, map_location);
  return Fsa(fsa_ptr);
}

}  // namespace sherpa
