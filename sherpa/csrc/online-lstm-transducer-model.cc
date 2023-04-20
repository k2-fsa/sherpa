// sherpa/csrc/online-lstm-transducer-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/online-lstm-transducer-model.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace sherpa {

OnlineLstmTransducerModel::OnlineLstmTransducerModel(
    const std::string &encoder_filename, const std::string &decoder_filename,
    const std::string &joiner_filename, torch::Device device /*=torch::kCPU*/)
    : device_(device) {
  encoder_ = torch::jit::load(encoder_filename, device);
  encoder_.eval();

  decoder_ = torch::jit::load(decoder_filename, device);
  encoder_.eval();

  joiner_ = torch::jit::load(joiner_filename, device);
  joiner_.eval();

  context_size_ =
      decoder_.attr("conv").toModule().attr("weight").toTensor().size(2);

  // Use 5 here since the subsampling is ((len - 3) // 2 - 1) // 2.
  int32_t pad_length = 5;

  chunk_shift_ = 4;
  chunk_size_ = chunk_shift_ + pad_length;
}

torch::IValue OnlineLstmTransducerModel::StateToIValue(const State &s) const {
  return torch::ivalue::Tuple::create(s.first, s.second);
}

OnlineLstmTransducerModel::State OnlineLstmTransducerModel::StateFromIValue(
    torch::IValue ivalue) const {
  // ivalue is a tuple containing two tensors
  auto tuple_ptr = ivalue.toTuple();

  torch::Tensor hidden_states = tuple_ptr->elements()[0].toTensor();
  torch::Tensor cell_states = tuple_ptr->elements()[1].toTensor();

  return {hidden_states, cell_states};
}

torch::IValue OnlineLstmTransducerModel::StackStates(
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

std::vector<torch::IValue> OnlineLstmTransducerModel::UnStackStates(
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

torch::IValue OnlineLstmTransducerModel::GetEncoderInitStates(
    int32_t batch_size /*=1*/) {
  torch::NoGradGuard no_grad;
  return encoder_.run_method("get_init_states", batch_size, device_);
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
OnlineLstmTransducerModel::RunEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor & /*num_processed_frames*/, torch::IValue states) {
  torch::NoGradGuard no_grad;

  // It returns [torch.Tensor, torch.Tensor, Pair[torch.Tensor, torch.Tensor]
  // which are [encoder_out, encoder_out_len, states]
  //
  // We skip the second entry `encoder_out_len` since we assume the
  // feature input is of fixed chunk size and there are no paddings.
  // We can figure out `encoder_out_len` from `encoder_out`.
  torch::IValue ivalue =
      encoder_.run_method("forward", features, features_length, states);
  auto tuple_ptr = ivalue.toTuple();
  torch::Tensor encoder_out = tuple_ptr->elements()[0].toTensor();

  torch::Tensor encoder_out_length = tuple_ptr->elements()[1].toTensor();

  auto next_states = tuple_ptr->elements()[2];

  return std::make_tuple(encoder_out, encoder_out_length, next_states);
}

torch::Tensor OnlineLstmTransducerModel::RunDecoder(
    const torch::Tensor &decoder_input) {
  torch::NoGradGuard no_grad;
  return decoder_
      .run_method("forward", decoder_input,
                  /*need_pad*/ torch::tensor({0}).to(torch::kBool))
      .toTensor();
}

torch::Tensor OnlineLstmTransducerModel::RunJoiner(
    const torch::Tensor &encoder_out, const torch::Tensor &decoder_out) {
  torch::NoGradGuard no_grad;
  return joiner_.run_method("forward", encoder_out, decoder_out).toTensor();
}

}  // namespace sherpa
