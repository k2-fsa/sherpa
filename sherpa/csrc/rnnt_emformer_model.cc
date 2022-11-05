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
#include "sherpa/csrc/rnnt_emformer_model.h"

#include <vector>

namespace sherpa {

RnntEmformerModel::RnntEmformerModel(const std::string &filename,
                                     torch::Device device /*=torch::kCPU*/,
                                     bool optimize_for_inference /*=false*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();

#if SHERPA_TORCH_VERSION_MAJOR > 1 || \
    (SHERPA_TORCH_VERSION_MAJOR == 1 && SHERPA_TORCH_VERSION_MINOR >= 10)
  // torch::jit::optimize_for_inference is available only in torch>=1.10
  if (optimize_for_inference) {
    model_ = torch::jit::optimize_for_inference(model_);
  }
#endif

  encoder_ = model_.attr("encoder").toModule();
  decoder_ = model_.attr("decoder").toModule();
  joiner_ = model_.attr("joiner").toModule();

  blank_id_ = decoder_.attr("blank_id").toInt();
  vocab_size_ = decoder_.attr("vocab_size").toInt();

  unk_id_ = blank_id_;
  if (decoder_.hasattr("unk_id")) {
    unk_id_ = decoder_.attr("unk_id").toInt();
  }

  context_size_ = decoder_.attr("context_size").toInt();
  segment_length_ = encoder_.attr("segment_length").toInt();
  right_context_length_ = encoder_.attr("right_context_length").toInt();
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
RnntEmformerModel::StreamingForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor & /*num_processed_frames*/, torch::IValue states) {
  torch::NoGradGuard no_grad;
  torch::IValue ivalue = encoder_.run_method("streaming_forward", features,
                                             features_length, states);
  auto tuple_ptr = ivalue.toTuple();
  torch::Tensor encoder_out = tuple_ptr->elements()[0].toTensor();
  torch::Tensor encoder_out_length = tuple_ptr->elements()[1].toTensor();
  torch::IValue next_states = tuple_ptr->elements()[2];

  return {encoder_out, encoder_out_length, next_states};
}

torch::IValue RnntEmformerModel::GetEncoderInitStates(int32_t /*unused=1*/) {
  return encoder_.run_method("get_init_state", device_);
}

torch::IValue RnntEmformerModel::StateToIValue(const State &states) const {
  torch::List<torch::List<torch::Tensor>> ans;
  ans.reserve(states.size());
  for (const auto &s : states) {
    ans.push_back(torch::List<torch::Tensor>{s});
  }
  return ans;
}

torch::IValue RnntEmformerModel::StackStates(
    const std::vector<torch::IValue> &ivalue) const {
  int32_t batch_size = ivalue.size();
  int32_t num_layers = 0;
  int32_t num_states = 0;

  // [layer][state][state_from_batch_i]
  std::vector<std::vector<std::vector<torch::Tensor>>> buf;
  for (const auto &v : ivalue) {
    std::vector<std::vector<torch::Tensor>> s = StateFromIValue(v);
    num_layers = s.size();
    if (buf.empty()) {
      buf.resize(num_layers);
    }

    for (int32_t layer = 0; layer != num_layers; ++layer) {
      const auto &layer_state = s[layer];
      num_states = layer_state.size();
      if (buf[layer].empty()) {
        buf[layer].resize(num_states);
      }

      for (int32_t n = 0; n != num_states; ++n) {
        if (buf[layer][n].empty()) {
          buf[layer][n].reserve(batch_size);
        }

        buf[layer][n].push_back(layer_state[n]);
      }
    }
  }

  State ans(num_layers);

  for (int32_t layer = 0; layer != num_layers; ++layer) {
    const auto &layer_state = buf[layer];
    ans[layer].reserve(num_states);
    for (const auto &s : layer_state) {
      auto stacked = torch::stack(s, /*dim*/ 1);
      ans[layer].push_back(stacked);
    }
  }

  return StateToIValue(ans);
}

std::vector<torch::IValue> RnntEmformerModel::UnStackStates(
    torch::IValue ivalue) const {
  auto states = StateFromIValue(ivalue);
  int32_t num_layers = states.size();
  int32_t batch_size = states[0][0].size(1);
  int32_t num_states = states[0].size();  // number of states per layer

  // [batch][layer][state]
  std::vector<std::vector<std::vector<torch::Tensor>>> buf(batch_size);
  for (auto &layer : buf) {
    layer.resize(num_layers);
    for (auto &s : layer) {
      s.reserve(num_states);
    }
  }

  for (int32_t layer = 0; layer != num_layers; ++layer) {
    const std::vector<torch::Tensor> &layer_state = states[layer];
    for (int32_t n = 0; n != num_states; ++n) {
      auto unstacked_state = torch::unbind(layer_state[n], /*dim*/ 1);
      for (int32_t b = 0; b != batch_size; ++b) {
        buf[b][layer].push_back(std::move(unstacked_state[b]));
      }
    }
  }

  std::vector<torch::IValue> ans(batch_size);
  for (int32_t b = 0; b != batch_size; ++b) {
    ans[b] = StateToIValue(buf[b]);
  }

  return ans;
}

RnntEmformerModel::State RnntEmformerModel::StateFromIValue(
    torch::IValue ivalue) const {
  torch::List<torch::IValue> list = ivalue.toList();

  int32_t num_layers = list.size();
  State ans;
  ans.reserve(num_layers);
  for (int32_t i = 0; i != num_layers; ++i) {
    ans.push_back(
        c10::impl::toTypedList<torch::Tensor>(list.get(i).toList()).vec());
  }
  return ans;
}

torch::Tensor RnntEmformerModel::ForwardDecoder(
    const torch::Tensor &decoder_input) {
  torch::NoGradGuard no_grad;
  return decoder_.run_method("forward", decoder_input, /*need_pad*/ false)
      .toTensor();
}

torch::Tensor RnntEmformerModel::ForwardJoiner(
    const torch::Tensor &encoder_out, const torch::Tensor &decoder_out) {
  torch::NoGradGuard no_grad;
  return joiner_.run_method("forward", encoder_out, decoder_out).toTensor();
}

}  // namespace sherpa
