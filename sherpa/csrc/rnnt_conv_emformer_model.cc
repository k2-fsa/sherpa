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
#include "sherpa/csrc/rnnt_conv_emformer_model.h"

#include <vector>

namespace sherpa {

RnntConvEmformerModel::RnntConvEmformerModel(
    const std::string &filename, torch::Device device /*=torch::kCPU*/,
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

  encoder_proj_ = joiner_.attr("encoder_proj").toModule();
  decoder_proj_ = joiner_.attr("decoder_proj").toModule();

  blank_id_ = decoder_.attr("blank_id").toInt();
  vocab_size_ = decoder_.attr("vocab_size").toInt();

  unk_id_ = blank_id_;
  if (decoder_.hasattr("unk_id")) {
    unk_id_ = decoder_.attr("unk_id").toInt();
  }

  context_size_ = decoder_.attr("context_size").toInt();
  chunk_length_ = encoder_.attr("chunk_length").toInt();
  auto right_context_length = encoder_.attr("right_context_length").toInt();
  // Add 2 here since we will drop the first and last frame after subsampling;
  // Add 3 here since the subsampling is ((len - 1) // 2 - 1) // 2.
  pad_length_ = right_context_length +
                2 * encoder_.attr("subsampling_factor").toInt() + 3;
}

torch::IValue RnntConvEmformerModel::StateToIValue(const State &s) const {
  return torch::ivalue::Tuple::create(s.first, s.second);
}

RnntConvEmformerModel::State RnntConvEmformerModel::StateFromIValue(
    torch::IValue ivalue) const {
  auto tuple_ptr_states = ivalue.toTuple();

  torch::List<torch::IValue> list_attn =
      tuple_ptr_states->elements()[0].toList();
  torch::List<torch::IValue> list_conv =
      tuple_ptr_states->elements()[1].toList();

  int32_t num_layers = list_attn.size();

  std::vector<std::vector<torch::Tensor>> next_state_attn;
  next_state_attn.reserve(num_layers);
  for (int32_t i = 0; i != num_layers; ++i) {
    next_state_attn.emplace_back(
        c10::impl::toTypedList<torch::Tensor>(list_attn.get(i).toList()).vec());
  }

  std::vector<torch::Tensor> next_state_conv;
  next_state_conv.reserve(num_layers);
  for (int32_t i = 0; i != num_layers; ++i) {
    next_state_conv.emplace_back(list_conv.get(i).toTensor());
  }

  return {next_state_attn, next_state_conv};
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
RnntConvEmformerModel::StreamingForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor &num_processed_frames, torch::IValue states) {
  torch::IValue ivalue = encoder_.run_method("infer", features, features_length,
                                             num_processed_frames, states);
  auto tuple_ptr = ivalue.toTuple();
  torch::Tensor encoder_out = tuple_ptr->elements()[0].toTensor();

  torch::Tensor encoder_out_length = tuple_ptr->elements()[1].toTensor();
  torch::IValue next_states = tuple_ptr->elements()[2];
  return {encoder_out, encoder_out_length, next_states};
}

torch::IValue RnntConvEmformerModel::GetEncoderInitStates() {
  return encoder_.run_method("init_states", device_);
}

torch::Tensor RnntConvEmformerModel::ForwardDecoder(
    const torch::Tensor &decoder_input) {
  return decoder_.run_method("forward", decoder_input, /*need_pad*/ false)
      .toTensor();
}

torch::Tensor RnntConvEmformerModel::ForwardJoiner(
    const torch::Tensor &projected_encoder_out,
    const torch::Tensor &projected_decoder_out) {
  return joiner_
      .run_method("forward", projected_encoder_out, projected_decoder_out,
                  /*project_input*/ false)
      .toTensor();
}

torch::Tensor RnntConvEmformerModel::ForwardEncoderProj(
    const torch::Tensor &encoder_out) {
  return encoder_proj_.run_method("forward", encoder_out).toTensor();
}

torch::Tensor RnntConvEmformerModel::ForwardDecoderProj(
    const torch::Tensor &decoder_out) {
  return decoder_proj_.run_method("forward", decoder_out).toTensor();
}
}  // namespace sherpa
