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

#include <fstream>

#include "gtest/gtest.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_stream.h"

namespace sherpa {

TEST(OnlineStream, Test) {
  float sampling_rate = 16000;
  int32_t feature_dim = 80;
  int32_t max_feature_vectors = 10;

  OnlineStream s(sampling_rate, feature_dim, max_feature_vectors);
  EXPECT_EQ(s.NumFramesReady(), 0);
  auto a = torch::rand({400}, torch::kFloat);
  s.AcceptWaveform(sampling_rate, a);

  EXPECT_EQ(s.NumFramesReady(), 1);
  auto frame = s.GetFrame(0);
  EXPECT_EQ(frame.dim(), 2);
  EXPECT_EQ(frame.size(0), 1);
  EXPECT_EQ(frame.size(1), feature_dim);

  EXPECT_FALSE(s.IsLastFrame(0));
  s.InputFinished();

  EXPECT_EQ(s.NumFramesReady(), 3);
  EXPECT_FALSE(s.IsLastFrame(1));
  EXPECT_TRUE(s.IsLastFrame(2));
}

}  // namespace sherpa
