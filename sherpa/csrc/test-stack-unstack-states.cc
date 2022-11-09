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
#include "sherpa/csrc/rnnt_emformer_model.h"
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

TEST(RnntEmformer, StackUnstackStates) {
  RnntEmformerModel model;
  int32_t num_layers = 12;
  int32_t memory_size = 2;
  int32_t input_dim = 10;
  int32_t left_context = 20;
  for (int32_t batch_size = 1; batch_size != 3; ++batch_size) {
    auto memory = torch::unbind(
        torch::rand({memory_size, batch_size, input_dim}, torch::kFloat), 1);

    auto key = torch::unbind(
        torch::rand({memory_size, batch_size, input_dim}, torch::kFloat), 1);

    auto value = torch::unbind(
        torch::rand({memory_size, batch_size, input_dim}, torch::kFloat), 1);

    auto past = torch::unbind(torch::rand({1, batch_size}, torch::kFloat), 1);

    std::vector<std::vector<std::vector<torch::Tensor>>> buf(batch_size);
    for (int32_t b = 0; b != batch_size; ++b) {
      auto &s = buf[b];
      s.resize(num_layers);
      for (int32_t layer = 0; layer != num_layers; ++layer) {
        auto &layer_states = s[layer];
        layer_states.push_back(memory[b]);
        layer_states.push_back(key[b]);
        layer_states.push_back(value[b]);
        layer_states.push_back(past[b]);
      }
    }

    std::vector<torch::IValue> states;
    for (const auto &s : buf) {
      states.push_back(model.StateToIValue(s));
    }

    auto stacked_states = model.StackStates(states);
    auto unstacked_states = model.UnStackStates(stacked_states);
    for (int32_t b = 0; b != batch_size; ++b) {
      auto target = model.StateFromIValue(unstacked_states[b]);
      // Check that s is identical to buf[b]
      const auto &ground_truth = buf[b];
      ASSERT_EQ(target.size(), ground_truth.size());
      for (int32_t layer = 0; layer != num_layers; ++layer) {
        const auto &t = target[layer];
        const auto &g = ground_truth[layer];
        ASSERT_EQ(t.size(), g.size());
        for (size_t i = 0; i != t.size(); ++i) {
          EXPECT_TRUE(torch::allclose(t[i], g[i]));
        }
      }
    }
  }
}

}  // namespace sherpa
