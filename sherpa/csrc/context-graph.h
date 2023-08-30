/**
 * Copyright      2023  Xiaomi Corporation (authors: Wei Kang)
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

#ifndef SHERPA_CSRC_CONTEXT_GRAPH_H_
#define SHERPA_CSRC_CONTEXT_GRAPH_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa/csrc/log.h"

namespace sherpa {

class ContextGraph;
using ContextGraphPtr = std::shared_ptr<ContextGraph>;

struct ContextState {
  int32_t token;
  float token_score;
  float node_score;
  float output_score;
  bool is_end;
  std::unordered_map<int32_t, std::unique_ptr<ContextState>> next;
  const ContextState *fail = nullptr;
  const ContextState *output = nullptr;

  ContextState() = default;
  ContextState(int32_t token, float token_score, float node_score,
               float output_score, bool is_end)
      : token(token),
        token_score(token_score),
        node_score(node_score),
        output_score(output_score),
        is_end(is_end) {}
};

class ContextGraph {
 public:
  ContextGraph() = default;
  ContextGraph(const std::vector<std::vector<int32_t>> &token_ids,
               float context_score)
      : context_score_(context_score) {
    root_ = std::make_unique<ContextState>(-1, 0, 0, 0, false);
    root_->fail = root_.get();
    Build(token_ids);
  }

  std::pair<float, const ContextState *> ForwardOneStep(
      const ContextState *state, int32_t token_id) const;
  std::pair<float, const ContextState *> Finalize(
      const ContextState *state) const;

  const ContextState *Root() const { return root_.get(); }

 private:
  float context_score_;
  std::unique_ptr<ContextState> root_;
  void Build(const std::vector<std::vector<int32_t>> &token_ids) const;
  void FillFailOutput() const;
};

}  // namespace sherpa
#endif  // SHERPA_CSRC_CONTEXT_GRAPH_H_
