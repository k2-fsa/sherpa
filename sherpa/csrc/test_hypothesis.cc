/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "gtest/gtest.h"
#include "sherpa/csrc/hypothesis.h"

namespace sherpa {

TEST(Hypothesis, DefaultConstructor) {
  Hypothesis hyp;
  EXPECT_TRUE(hyp.ys.empty());
  EXPECT_EQ(hyp.log_prob, 0);
}

TEST(Hypothesis, Constructor) {
  Hypothesis hyp({1, 2, 3}, 0.5);
  EXPECT_EQ(hyp.ys, (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(hyp.log_prob, 0.5);
}

TEST(Hypothesis, Key) {
  Hypothesis hyp;
  hyp.ys = {1, 2, 3};
  EXPECT_EQ(hyp.Key(), "1-2-3");
}

TEST(Hypotheses, ConstructorFromVector) {
  std::vector<Hypothesis> hyp_vec;
  hyp_vec.emplace_back(Hypothesis({1, 2, 3}, -1.5));
  hyp_vec.emplace_back(Hypothesis({30}, -2.5));

  EXPECT_EQ(hyp_vec[0].ys.size(), 3);
  EXPECT_EQ(hyp_vec[1].ys.size(), 1);

  Hypotheses hyps(std::move(hyp_vec));
  EXPECT_EQ(hyps.Size(), 2);

  EXPECT_EQ(hyp_vec.size(), 0);
}

}  // namespace sherpa
