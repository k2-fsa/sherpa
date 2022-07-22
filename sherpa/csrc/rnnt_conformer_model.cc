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

  encoder_proj_ = joiner_.attr("encoder_proj").toModule();
  decoder_proj_ = joiner_.attr("decoder_proj").toModule();

  blank_id_ = decoder_.attr("blank_id").toInt();
  vocab_size_ = decoder_.attr("vocab_size").toInt();

  unk_id_ = blank_id_;
  if (decoder_.hasattr("unk_id")) {
    unk_id_ = decoder_.attr("unk_id").toInt();
  }

  context_size_ = decoder_.attr("context_size").toInt();
}

std::pair<torch::Tensor, torch::Tensor> RnntConformerModel::ForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length) {
  auto outputs = model_.attr("encoder")
                     .toModule()
                     .run_method("forward", features, features_length)
                     .toTuple();

  auto encoder_out = outputs->elements()[0].toTensor();
  auto encoder_out_length = outputs->elements()[1].toTensor();

  return {encoder_out, encoder_out_length};
}

RnntConformerModel::State RnntConformerModel::GetEncoderInitStates(
    int32_t left_context) {
  torch::IValue ivalue =
      encoder_.run_method("get_init_state", left_context, device_);
  torch::List<torch::IValue> list = ivalue.toList();

  RnntConformerModel::State states = {list.get(0).toTensor(),
                                      list.get(1).toTensor()};
  return states;
}

std::tuple<torch::Tensor, torch::Tensor, RnntConformerModel::State>
RnntConformerModel::StreamingForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const RnntConformerModel::State &states,
    const torch::Tensor &processed_frames, int32_t left_context,
    int32_t right_context) {
  auto outputs =
      encoder_
          .run_method("streaming_forward", features, features_length, states,
                      processed_frames, left_context, right_context)
          .toTuple();
  auto encoder_out = outputs->elements()[0].toTensor();
  auto encoder_out_length = outputs->elements()[1].toTensor();

  torch::List<torch::IValue> list = outputs->elements()[2].toList();

  RnntConformerModel::State next_states = {list.get(0).toTensor(),
                                           list.get(1).toTensor()};

  return {encoder_out, encoder_out_length, next_states};
}

torch::Tensor RnntConformerModel::ForwardDecoder(
    const torch::Tensor &decoder_input) {
  return decoder_.run_method("forward", decoder_input, /*need_pad*/ false)
      .toTensor();
}

torch::Tensor RnntConformerModel::ForwardJoiner(
    const torch::Tensor &projected_encoder_out,
    const torch::Tensor &projected_decoder_out) {
  return joiner_
      .run_method("forward", projected_encoder_out, projected_decoder_out,
                  /*project_input*/ false)
      .toTensor();
}

torch::Tensor RnntConformerModel::ForwardEncoderProj(
    const torch::Tensor &encoder_out) {
  return encoder_proj_.run_method("forward", encoder_out).toTensor();
}

torch::Tensor RnntConformerModel::ForwardDecoderProj(
    const torch::Tensor &decoder_out) {
  return decoder_proj_.run_method("forward", decoder_out).toTensor();
}

}  // namespace sherpa
