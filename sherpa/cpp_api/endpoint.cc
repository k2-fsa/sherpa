/**
 * Copyright      2022  (authors: Pingfeng Luo)
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
#include "sherpa/cpp_api/endpoint.h"

#include <string>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

static bool RuleActivated(const EndpointRule &rule,
                          const std::string &rule_name,
                          const float trailing_silence,
                          const float utterance_length) {
  bool contain_nonsilence = utterance_length > trailing_silence;
  bool ans = (contain_nonsilence || !rule.must_contain_nonsilence) &&
             trailing_silence >= rule.min_trailing_silence &&
             utterance_length >= rule.min_utterance_length;
  if (ans) {
    SHERPA_LOG(DEBUG) << "Endpointing rule " << rule_name << " activated: "
                      << (contain_nonsilence ? "true" : "false") << ','
                      << trailing_silence << ',' << utterance_length;
  }
  return ans;
}

static void RegisterEndpointRule(ParseOptions *po, EndpointRule *rule,
                                 const std::string &rule_name) {
  po->Register(
      rule_name + "-must-contain-nonsilence", &rule->must_contain_nonsilence,
      "If True, for this endpointing " + rule_name +
          " to apply there must"
          "be nonsilence in the best-path traceback."
          "For decoding, a non-blank token is considered as non-silence");
  po->Register(rule_name + "-min-trailing-silence", &rule->min_trailing_silence,
               "This endpointing " + rule_name +
                   " requires duration of trailing silence"
                   "(in seconds) to be >= this value.");
  po->Register(rule_name + "-min-utterance-length", &rule->min_utterance_length,
               "This endpointing " + rule_name +
                   " requires utterance-length (in seconds)"
                   "to be >= this value.");
}

void EndpointConfig::Register(ParseOptions *po) {
  RegisterEndpointRule(po, &rule1, "rule1");
  RegisterEndpointRule(po, &rule2, "rule2");
  RegisterEndpointRule(po, &rule3, "rule3");
}

bool Endpoint::IsEndpoint(const int num_frames_decoded,
                          const int trailing_silence_frames,
                          const float frame_shift_in_seconds) const {
  float utterance_length = num_frames_decoded * frame_shift_in_seconds;
  float trailing_silence = trailing_silence_frames * frame_shift_in_seconds;
  if (RuleActivated(config_.rule1, "rule1", trailing_silence,
                    utterance_length) ||
      RuleActivated(config_.rule1, "rule2", trailing_silence,
                    utterance_length) ||
      RuleActivated(config_.rule3, "rule3", trailing_silence,
                    utterance_length)) {
    return true;
  }
  return false;
}

}  // namespace sherpa
