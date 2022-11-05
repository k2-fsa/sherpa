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
