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
#include "sherpa/python/csrc/resample.h"
//
#include "sherpa/python/csrc/endpoint.h"
#include "sherpa/python/csrc/fast-beam-search-config.h"
#include "sherpa/python/csrc/feature-config.h"
#include "sherpa/python/csrc/offline-ctc-model.h"
#include "sherpa/python/csrc/offline-recognizer.h"
#include "sherpa/python/csrc/offline-stream.h"
#include "sherpa/python/csrc/online-recognizer.h"
#include "sherpa/python/csrc/online-stream.h"

namespace sherpa {

PYBIND11_MODULE(_sherpa, m) {
  m.doc() = "pybind11 binding of sherpa";
  m.attr("cxx_flags") = std::string(kCMakeCxxFlags);

  PybindResample(m);

  PybindFeatureConfig(m);
  PybindFastBeamSearch(m);
  PybindOfflineCtcModel(m);
  PybindOfflineStream(m);
  PybindOfflineRecognizer(m);
  PybindEndpoint(m);
  PybindOnlineStream(m);
  PybindOnlineRecognizer(m);
}

}  // namespace sherpa
