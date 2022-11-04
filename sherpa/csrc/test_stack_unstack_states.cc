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
#include "sherpa/csrc/rnnt_conformer_model.h"
#include "sherpa/csrc/rnnt_lstm_model.h"

namespace sherpa {

TEST(RnntLstmModel, StackUnstackStates) {
  RnntLstmModel model;

  int32_t num_layers = 12;
  int32_t proj_size = 512;
  int32_t hidden_size = 2048;

  torch::Tensor hx0 = torch::rand({num_layers, 1, proj_size}, torch::kFloat);
  torch::Tensor cx0 = torch::rand({num_layers, 1, proj_size}, torch::kFloat);

  torch::Tensor hx1 = torch::rand({num_layers, 1, proj_size}, torch::kFloat);
  torch::Tensor cx1 = torch::rand({num_layers, 1, proj_size}, torch::kFloat);

  torch::Tensor hx2 = torch::rand({num_layers, 1, proj_size}, torch::kFloat);
  torch::Tensor cx2 = torch::rand({num_layers, 1, proj_size}, torch::kFloat);

  torch::IValue s0 = model.StateToIValue({hx0, cx0});
  torch::IValue s1 = model.StateToIValue({hx1, cx1});
  torch::IValue s2 = model.StateToIValue({hx2, cx2});

  {
    // Test batch size 1
    auto stacked = model.StackStates({s0});
    auto unstacked = model.UnStackStates(stacked);
    EXPECT_EQ(unstacked.size(), 1);
    auto state = model.StateFromIValue(unstacked[0]);
    EXPECT_TRUE(torch::allclose(state.first, hx0));
    EXPECT_TRUE(torch::allclose(state.second, cx0));
  }

  {
    // Test batch size 2
    auto stacked = model.StackStates({s0, s1});
    auto unstacked = model.UnStackStates(stacked);
    EXPECT_EQ(unstacked.size(), 2);
    auto state = model.StateFromIValue(unstacked[0]);
    EXPECT_TRUE(torch::allclose(state.first, hx0));
    EXPECT_TRUE(torch::allclose(state.second, cx0));

    state = model.StateFromIValue(unstacked[1]);
    EXPECT_TRUE(torch::allclose(state.first, hx1));
    EXPECT_TRUE(torch::allclose(state.second, cx1));
  }

  {
    // Test batch size 3
    auto stacked = model.StackStates({s0, s1, s2});
    auto unstacked = model.UnStackStates(stacked);
    EXPECT_EQ(unstacked.size(), 3);
    auto state = model.StateFromIValue(unstacked[0]);
    EXPECT_TRUE(torch::allclose(state.first, hx0));
    EXPECT_TRUE(torch::allclose(state.second, cx0));

    state = model.StateFromIValue(unstacked[1]);
    EXPECT_TRUE(torch::allclose(state.first, hx1));
    EXPECT_TRUE(torch::allclose(state.second, cx1));

    state = model.StateFromIValue(unstacked[2]);
    EXPECT_TRUE(torch::allclose(state.first, hx2));
    EXPECT_TRUE(torch::allclose(state.second, cx2));
  }
}

TEST(RnntConformerModel, StackUnstackStates) {
  RnntConformerModel model;
  int32_t num_layers = 12;
  int32_t left_context = 32;
  int32_t encoder_dim = 512;
  int32_t cnn_module_kernel = 31;

  torch::Tensor attn0 =
      torch::rand({num_layers, left_context, encoder_dim}, torch::kFloat);
  torch::Tensor conv0 = torch::rand(
      {num_layers, cnn_module_kernel - 1, encoder_dim}, torch::kFloat);

  torch::Tensor attn1 =
      torch::rand({num_layers, left_context, encoder_dim}, torch::kFloat);
  torch::Tensor conv1 = torch::rand(
      {num_layers, cnn_module_kernel - 1, encoder_dim}, torch::kFloat);

  torch::Tensor attn2 =
      torch::rand({num_layers, left_context, encoder_dim}, torch::kFloat);
  torch::Tensor conv2 = torch::rand(
      {num_layers, cnn_module_kernel - 1, encoder_dim}, torch::kFloat);

  torch::IValue s0 = model.StateToIValue({attn0, conv0});
  torch::IValue s1 = model.StateToIValue({attn1, conv1});
  torch::IValue s2 = model.StateToIValue({attn2, conv2});

  {
    // Test batch size 1
    auto stacked = model.StackStates({s0});
    auto unstacked = model.UnStackStates(stacked);
    EXPECT_EQ(unstacked.size(), 1);
    auto state = model.StateFromIValue(unstacked[0]);
    EXPECT_TRUE(torch::allclose(state[0], attn0));
    EXPECT_TRUE(torch::allclose(state[1], conv0));
  }

  {
    // Test batch size 2
    auto stacked = model.StackStates({s0, s1});
    auto unstacked = model.UnStackStates(stacked);
    EXPECT_EQ(unstacked.size(), 2);

    auto state = model.StateFromIValue(unstacked[0]);
    EXPECT_TRUE(torch::allclose(state[0], attn0));
    EXPECT_TRUE(torch::allclose(state[1], conv0));

    state = model.StateFromIValue(unstacked[1]);
    EXPECT_TRUE(torch::allclose(state[0], attn1));
    EXPECT_TRUE(torch::allclose(state[1], conv1));
  }

  {
    // Test batch size 3
    auto stacked = model.StackStates({s0, s1, s2});
    auto unstacked = model.UnStackStates(stacked);
    EXPECT_EQ(unstacked.size(), 3);
    auto state = model.StateFromIValue(unstacked[0]);
    EXPECT_TRUE(torch::allclose(state[0], attn0));
    EXPECT_TRUE(torch::allclose(state[1], conv0));

    state = model.StateFromIValue(unstacked[1]);
    EXPECT_TRUE(torch::allclose(state[0], attn1));
    EXPECT_TRUE(torch::allclose(state[1], conv1));

    state = model.StateFromIValue(unstacked[2]);
    EXPECT_TRUE(torch::allclose(state[0], attn2));
    EXPECT_TRUE(torch::allclose(state[1], conv2));
  }
}

}  // namespace sherpa
