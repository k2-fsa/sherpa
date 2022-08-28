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

#include <string>

#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_stream.h"
#include "torch/script.h"

static void CheckStates(torch::IValue a, torch::IValue b) {
  TORCH_CHECK(a.tagKind() == b.tagKind(), a.tagKind(), " vs ", b.tagKind());

  auto a_tuple_ptr = a.toTuple();
  torch::List<torch::IValue> a_list_attn = a_tuple_ptr->elements()[0].toList();
  torch::List<torch::IValue> a_list_conv = a_tuple_ptr->elements()[1].toList();

  auto b_tuple_ptr = b.toTuple();
  torch::List<torch::IValue> b_list_attn = b_tuple_ptr->elements()[0].toList();
  torch::List<torch::IValue> b_list_conv = b_tuple_ptr->elements()[1].toList();

  int32_t num_layers = a_list_attn.size();
  TORCH_CHECK(num_layers == b_list_attn.size());
  for (int32_t i = 0; i != num_layers; ++i) {
    torch::IValue a_i = a_list_attn[i];
    torch::IValue b_i = b_list_attn[i];

    // for attn
    auto a_i_v = c10::impl::toTypedList<torch::Tensor>(a_i.toList()).vec();
    auto b_i_v = c10::impl::toTypedList<torch::Tensor>(b_i.toList()).vec();
    for (size_t k = 0; k != a_i_v.size(); ++k) {
      TORCH_CHECK(torch::equal(a_i_v[k], b_i_v[k]));
    }

    // for conv
    auto a_t = static_cast<const torch::IValue &>(a_list_conv[i]).toTensor();
    auto b_t = static_cast<const torch::IValue &>(b_list_conv[i]).toTensor();

    TORCH_CHECK(torch::equal(a_t, b_t));
  }  // for (int32_t i = 0; i != num_layers; ++i)
}

int main() {
  float sampling_rate = 16000;
  int32_t feature_dim = 80;
  int32_t max_feature_vectors = 10;

  torch::Device device(torch::kCPU);

  sherpa::OnlineStream s(sampling_rate, feature_dim, max_feature_vectors);

  std::string nn_model = "./cpu-conv-emformer-jit.pt";

  torch::jit::Module model = torch::jit::load(nn_model, device);

  torch::jit::Module encoder = model.attr("encoder").toModule();

  // Tuple[List[List[torch.Tensor]], List[torch.Tensor]]
  torch::IValue states0 = encoder.run_method("init_states", device);
  torch::IValue states1 = encoder.run_method("init_states", device);
  torch::IValue states2 = encoder.run_method("init_states", device);

  torch::IValue stacked_states = s.StackStates({states0, states1, states2});
  std::vector<torch::IValue> states = s.UnStackStates(stacked_states);
  TORCH_CHECK(states.size() == 3);

  CheckStates(states0, states[0]);
  CheckStates(states1, states[1]);
  CheckStates(states2, states[2]);

  return 0;
}
