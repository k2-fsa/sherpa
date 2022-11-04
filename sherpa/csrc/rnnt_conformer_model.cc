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
#include "sherpa/csrc/rnnt_conformer_model.h"

#include <tuple>

namespace sherpa {

RnntConformerModel::RnntConformerModel(const std::string &filename,
                                       int32_t left_context,
                                       int32_t right_context,
                                       int32_t decode_chunk_size,
                                       torch::Device device /*=torch::kCPU*/,
                                       bool optimize_for_inference /*=false*/)
    : device_(device),
      left_context_(left_context),
      right_context_(right_context) {
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

  // We add 3 here since the subsampling method is using
  // ((len - 1) // 2 - 1) // 2)
  // We plus 2 here because we will cut off one frame on each side
  // of encoder_embed output (in conformer.py) to avoid a training
  // and decoding mismatch by seeing padding values.
  // Note: chunk_length is in frames before subsampling.
  //
  // (decode_chunk_size + 2 + right_context_) * subsampling_factor_ + 3;

  chunk_length_ = decode_chunk_size * subsampling_factor_;
  pad_length_ = (2 + right_context_) * subsampling_factor_ + 3;
}

std::pair<torch::Tensor, torch::Tensor> RnntConformerModel::ForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length) {
  torch::NoGradGuard no_grad;

  auto outputs = model_.attr("encoder")
                     .toModule()
                     .run_method("forward", features, features_length)
                     .toTuple();

  auto encoder_out = outputs->elements()[0].toTensor();
  auto encoder_out_length = outputs->elements()[1].toTensor();

  return {encoder_out, encoder_out_length};
}

torch::IValue RnntConformerModel::StateToIValue(const State &s) const {
  return torch::IValue(s);
}

RnntConformerModel::State RnntConformerModel::StateFromIValue(
    torch::IValue ivalue) const {
  torch::List<torch::IValue> list = ivalue.toList();

  return {list.get(0).toTensor(), list.get(1).toTensor()};
}

torch::IValue RnntConformerModel::StackStates(
    const std::vector<torch::IValue> &states) const {
  int32_t batch_size = states.size();
  std::vector<torch::Tensor> attn;
  std::vector<torch::Tensor> conv;
  attn.reserve(batch_size);
  conv.reserve(batch_size);

  for (const auto &s : states) {
    torch::List<torch::IValue> list = s.toList();
    attn.push_back(list.get(0).toTensor());
    conv.push_back(list.get(1).toTensor());
  }
  torch::Tensor stacked_attn = torch::stack(attn, /*dim*/ 2);
  torch::Tensor stacked_conv = torch::stack(conv, /*dim*/ 2);

  return torch::List<torch::Tensor>({stacked_attn, stacked_conv});
}

std::vector<torch::IValue> RnntConformerModel::UnStackStates(
    torch::IValue ivalue) const {
  State states = StateFromIValue(ivalue);
  int32_t batch_size = states[0].size(2);
  std::vector<torch::IValue> ans;
  ans.reserve(batch_size);

  auto stacked_attn = torch::unbind(states[0], /*dim*/ 2);
  auto stacked_conv = torch::unbind(states[1], /*dim*/ 2);
  for (int32_t i = 0; i != batch_size; ++i) {
    auto attn = stacked_attn[i];
    auto conv = stacked_conv[i];
    ans.push_back(StateToIValue({attn, conv}));
  }

  return ans;
}

torch::IValue RnntConformerModel::GetEncoderInitStates(int32_t /*unused=1*/) {
  torch::NoGradGuard no_grad;
  return encoder_.run_method("get_init_state", left_context_, device_);
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
RnntConformerModel::StreamingForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor &processed_frames, torch::IValue states) {
  torch::NoGradGuard no_grad;
  auto outputs =
      encoder_
          .run_method("streaming_forward", features, features_length, states,
                      processed_frames, left_context_, right_context_)
          .toTuple();
  auto encoder_out = outputs->elements()[0].toTensor();
  auto encoder_out_length = outputs->elements()[1].toTensor();

  auto next_states = outputs->elements()[2];

  return {encoder_out, encoder_out_length, next_states};
}

torch::Tensor RnntConformerModel::ForwardDecoder(
    const torch::Tensor &decoder_input) {
  torch::NoGradGuard no_grad;
  return decoder_.run_method("forward", decoder_input, /*need_pad*/ false)
      .toTensor();
}

torch::Tensor RnntConformerModel::ForwardJoiner(
    const torch::Tensor &projected_encoder_out,
    const torch::Tensor &projected_decoder_out) {
  torch::NoGradGuard no_grad;
  return joiner_
      .run_method("forward", projected_encoder_out, projected_decoder_out,
                  /*project_input*/ false)
      .toTensor();
}

torch::Tensor RnntConformerModel::ForwardEncoderProj(
    const torch::Tensor &encoder_out) {
  torch::NoGradGuard no_grad;
  return encoder_proj_.run_method("forward", encoder_out).toTensor();
}

torch::Tensor RnntConformerModel::ForwardDecoderProj(
    const torch::Tensor &decoder_out) {
  torch::NoGradGuard no_grad;
  return decoder_proj_.run_method("forward", decoder_out).toTensor();
}

}  // namespace sherpa
