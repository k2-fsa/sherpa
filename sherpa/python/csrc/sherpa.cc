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

#include "sherpa/python/csrc/sherpa.h"

#include <string>

#include "sherpa/csrc/version.h"
#include "sherpa/python/csrc/hypothesis.h"
#include "sherpa/python/csrc/rnnt_beam_search.h"
#include "sherpa/python/csrc/rnnt_conformer_model.h"
#include "sherpa/python/csrc/rnnt_conv_emformer_model.h"
#include "sherpa/python/csrc/rnnt_emformer_model.h"
#include "sherpa/python/csrc/rnnt_model.h"

namespace sherpa {

PYBIND11_MODULE(_sherpa, m) {
  m.doc() = "pybind11 binding of sherpa";
  m.attr("cxx_flags") = std::string(kCMakeCxxFlags);

  PybindHypothesis(m);
  PybindRnntModel(m);
  PybindRnntConformerModel(m);
  PybindRnntConvEmformerModel(m);
  PybindRnntEmformerModel(m);
  PybindRnntBeamSearch(m);
}

}  // namespace sherpa
