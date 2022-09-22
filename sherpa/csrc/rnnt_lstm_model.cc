/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang,
 *                                                   Zengwei Yao)
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
#include "sherpa/csrc/rnnt_lstm_model.h"

#include <memory>
#include <vector>

namespace sherpa {

RnntLstmModel::RnntLstmModel(const std::string &encoder_filename,
                             const std::string &decoder_filename,
                             const std::string &joiner_filename,
                             torch::Device device /*=torch::kCPU*/,
                             bool optimize_for_inference /*=false*/)
    : device_(device) {
  encoder_ = torch::jit::load(encoder_filename, device);
  encoder_.eval();
  decoder_ = torch::jit::load(decoder_filename, device);
  encoder_.eval();
  joiner_ = torch::jit::load(joiner_filename, device);
  joiner_.eval();

#if SHERPA_TORCH_VERSION_MAJOR > 1 || \
    (SHERPA_TORCH_VERSION_MAJOR == 1 && SHERPA_TORCH_VERSION_MINOR >= 10)
  // torch::jit::optimize_for_inference is available only in torch>=1.10
  if (optimize_for_inference) {
    encoder_ = torch::jit::optimize_for_inference(encoder_);
    decoder_ = torch::jit::optimize_for_inference(decoder_);
    joiner_ = torch::jit::optimize_for_inference(joiner_);
  }
#endif

  vocab_size_ = joiner_.attr("output_linear")
                    .toModule()
                    .attr("weight")
                    .toTensor()
                    .size(0);
  context_size_ =
      decoder_.attr("conv").toModule().attr("weight").toTensor().size(2);

  // hard code following attributes
  blank_id_ = 0;
  unk_id_ = blank_id_;
  // Add 5 here since the subsampling is ((len - 3) // 2 - 1) // 2.
  pad_length_ = 5;
  subsampling_factor_ = 4;
}

std::tuple<torch::Tensor, torch::Tensor, RnntLstmModel::State>
RnntLstmModel::StreamingForwardEncoder(const torch::Tensor &features,
                                       const torch::Tensor &features_length,
                                       State states) {
  // It contains [torch.Tensor, torch.Tensor, Pair[torch.Tensor, torch.Tensor]
  // which are [encoder_out, encoder_out_len, states]
  //
  // We skip the second entry `encoder_out_len` since we assume the
  // feature input are of fixed chunk size and there are no paddings.
  // We can figure out `encoder_out_len` from `encoder_out`.
  auto states_tuple = torch::ivalue::Tuple::create(states.first, states.second);
  torch::IValue ivalue =
      encoder_.run_method("forward", features, features_length, states_tuple);
  auto tuple_ptr = ivalue.toTuple();
  torch::Tensor encoder_out = tuple_ptr->elements()[0].toTensor();

  torch::Tensor encoder_out_length = tuple_ptr->elements()[1].toTensor();

  auto tuple_ptr_states = tuple_ptr->elements()[2].toTuple();
  torch::Tensor hidden_states = tuple_ptr_states->elements()[0].toTensor();
  torch::Tensor cell_states = tuple_ptr_states->elements()[1].toTensor();
  State next_states = {hidden_states, cell_states};

  return {encoder_out, encoder_out_length, next_states};
}

RnntLstmModel::State RnntLstmModel::GetEncoderInitStates(
    int32_t batch_size /*=1*/) {
  torch::IValue ivalue =
      encoder_.run_method("get_init_states", batch_size, device_);
  auto tuple_ptr = ivalue.toTuple();

  torch::Tensor hidden_states = tuple_ptr->elements()[0].toTensor();
  torch::Tensor cell_states = tuple_ptr->elements()[1].toTensor();

  return {hidden_states, cell_states};
}  // namespace sherpa

torch::Tensor RnntLstmModel::ForwardDecoder(
    const torch::Tensor &decoder_input) {
  return decoder_
      .run_method("forward", decoder_input,
                  /*need_pad*/ torch::tensor({0}).to(torch::kBool))
      .toTensor();
}

torch::Tensor RnntLstmModel::ForwardJoiner(const torch::Tensor &encoder_out,
                                           const torch::Tensor &decoder_out) {
  return joiner_.run_method("forward", encoder_out, decoder_out).toTensor();
}

}  // namespace sherpa
