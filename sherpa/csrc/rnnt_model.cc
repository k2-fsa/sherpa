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
#include "sherpa/csrc/rnnt_model.h"

namespace sherpa {

RnntModel::RnntModel(const std::string &filename,
                     torch::Device device /*=torch::kCPU*/,
                     bool optimize_for_inference /*=false*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();
  if (optimize_for_inference) {
    model_ = torch::jit::optimize_for_inference(model_);
  }

  encoder_ = model_.attr("encoder").toModule();
  decoder_ = model_.attr("decoder").toModule();
  joiner_ = model_.attr("joiner").toModule();

  encoder_proj_ = joiner_.attr("encoder_proj").toModule();
  decoder_proj_ = joiner_.attr("decoder_proj").toModule();

  blank_id_ = decoder_.attr("blank_id").toInt();

  unk_id_ = blank_id_;
  if (decoder_.hasattr("unk_id")) {
    unk_id_ = decoder_.attr("unk_id").toInt();
  }

  context_size_ = decoder_.attr("context_size").toInt();
}

std::pair<torch::Tensor, torch::Tensor> RnntModel::ForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length) {
  auto outputs = model_.attr("encoder")
                     .toModule()
                     .run_method("forward", features, features_length)
                     .toTuple();

  auto encoder_out = outputs->elements()[0].toTensor();
  auto encoder_out_length = outputs->elements()[1].toTensor();

  return {encoder_out, encoder_out_length};
}

torch::Tensor RnntModel::ForwardDecoder(const torch::Tensor &decoder_input) {
  return decoder_.run_method("forward", decoder_input, /*need_pad*/ false)
      .toTensor();
}

torch::Tensor RnntModel::ForwardJoiner(
    const torch::Tensor &projected_encoder_out,
    const torch::Tensor &projected_decoder_out) {
  return joiner_
      .run_method("forward", projected_encoder_out, projected_decoder_out,
                  /*project_input*/ false)
      .toTensor();
}

torch::Tensor RnntModel::ForwardEncoderProj(const torch::Tensor &encoder_out) {
  return encoder_proj_.run_method("forward", encoder_out).toTensor();
}

torch::Tensor RnntModel::ForwardDecoderProj(const torch::Tensor &decoder_out) {
  return decoder_proj_.run_method("forward", decoder_out).toTensor();
}

}  // namespace sherpa
