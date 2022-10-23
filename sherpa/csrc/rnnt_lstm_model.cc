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
  chunk_length_ = 4;
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
RnntLstmModel::StreamingForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor & /*unused_num_processed_frames*/,
    torch::IValue states) {
  // It contains [torch.Tensor, torch.Tensor, Pair[torch.Tensor, torch.Tensor]
  // which are [encoder_out, encoder_out_len, states]
  //
  // We skip the second entry `encoder_out_len` since we assume the
  // feature input are of fixed chunk size and there are no paddings.
  // We can figure out `encoder_out_len` from `encoder_out`.
  torch::IValue ivalue =
      encoder_.run_method("forward", features, features_length, states);
  auto tuple_ptr = ivalue.toTuple();
  torch::Tensor encoder_out = tuple_ptr->elements()[0].toTensor();

  torch::Tensor encoder_out_length = tuple_ptr->elements()[1].toTensor();

  auto next_states = tuple_ptr->elements()[2];

  return {encoder_out, encoder_out_length, next_states};
}

torch::IValue RnntLstmModel::StateToIValue(const State &s) const {
  return torch::ivalue::Tuple::create(s.first, s.second);
}

RnntLstmModel::State RnntLstmModel::StateFromIValue(
    torch::IValue ivalue) const {
  // ivalue is a tuple containing two tensors
  auto tuple_ptr = ivalue.toTuple();

  torch::Tensor hidden_states = tuple_ptr->elements()[0].toTensor();
  torch::Tensor cell_states = tuple_ptr->elements()[1].toTensor();

  return {hidden_states, cell_states};
}

torch::IValue RnntLstmModel::GetEncoderInitStates(int32_t batch_size /*=1*/) {
  return encoder_.run_method("get_init_states", batch_size, device_);
}

torch::IValue RnntLstmModel::StackStates(
    const std::vector<torch::IValue> &states) const {
  auto n = static_cast<int32_t>(states.size());

  std::vector<torch::Tensor> hx;
  std::vector<torch::Tensor> cx;

  hx.reserve(n);
  cx.reserve(n);
  for (const auto &ivalue : states) {
    auto s = StateFromIValue(ivalue);
    hx.push_back(std::move(s.first));
    cx.push_back(std::move(s.second));
  }

  auto cat_hx = torch::cat(hx, /*dim*/ 1);
  auto cat_cx = torch::cat(cx, /*dim*/ 1);

  return torch::ivalue::Tuple::create(cat_hx, cat_cx);
}

std::vector<torch::IValue> RnntLstmModel::UnStackStates(
    torch::IValue ivalue) const {
  auto states = StateFromIValue(ivalue);

  std::vector<torch::Tensor> hx = states.first.unbind(/*dim*/ 1);
  std::vector<torch::Tensor> cx = states.second.unbind(/*dim*/ 1);
  auto n = static_cast<int32_t>(hx.size());

  std::vector<torch::IValue> ans(n);
  for (int32_t i = 0; i != n; ++i) {
    auto h = hx[i].unsqueeze(/*dim*/ 1);
    auto c = cx[i].unsqueeze(/*dim*/ 1);
    ans[i] = torch::ivalue::Tuple::create(h, c);
  }

  return ans;
}

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
