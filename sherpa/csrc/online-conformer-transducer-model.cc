// sherpa/csrc/online-conformer-transducer-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/online-conformer-transducer-model.h"

#include <string>
#include <tuple>
#include <vector>

namespace sherpa {

OnlineConformerTransducerModel::OnlineConformerTransducerModel(
    const std::string &filename, int32_t left_context, int32_t right_context,
    int32_t decode_chunk_size, torch::Device device /*= torch::kCPU*/)
    : device_(device),
      left_context_(left_context),
      right_context_(right_context) {
  model_ = torch::jit::load(filename, device);
  model_.eval();

  encoder_ = model_.attr("encoder").toModule();
  decoder_ = model_.attr("decoder").toModule();
  joiner_ = model_.attr("joiner").toModule();

  encoder_proj_ = joiner_.attr("encoder_proj").toModule();
  decoder_proj_ = joiner_.attr("decoder_proj").toModule();

  int32_t subsampling_factor = encoder_.attr("subsampling_factor").toInt();

  context_size_ = decoder_.attr("context_size").toInt();

  // We add 3 here since the subsampling method is using
  // ((len - 1) // 2 - 1) // 2)
  // We plus 2 here because we will cut off one frame on each side
  // of encoder_embed output (in conformer.py) to avoid a training
  // and decoding mismatch by seeing padding values.
  int32_t pad_length =
      2 * subsampling_factor + right_context + (subsampling_factor - 1);
  chunk_shift_ = decode_chunk_size;
  chunk_size_ = chunk_shift_ + pad_length;
  // Note: Differences from the conv-emformer:
  //  right_context in streaming conformer is specified by users during
  //  decoding and it is a value before subsampling.
}

torch::IValue OnlineConformerTransducerModel::StateToIValue(
    const State &s) const {
  return torch::IValue(s);
}

OnlineConformerTransducerModel::State
OnlineConformerTransducerModel::StateFromIValue(torch::IValue ivalue) const {
  torch::List<torch::IValue> list = ivalue.toList();

  return {list.get(0).toTensor(), list.get(1).toTensor()};
}

torch::IValue OnlineConformerTransducerModel::StackStates(
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

std::vector<torch::IValue> OnlineConformerTransducerModel::UnStackStates(
    torch::IValue ivalue) const {
  State states = StateFromIValue(ivalue);
  int32_t batch_size = states[0].size(2);
  std::vector<torch::IValue> ans;
  ans.reserve(batch_size);

  auto unstacked_attn = torch::unbind(states[0], /*dim*/ 2);
  auto unstacked_conv = torch::unbind(states[1], /*dim*/ 2);
  for (int32_t i = 0; i != batch_size; ++i) {
    auto attn = unstacked_attn[i];
    auto conv = unstacked_conv[i];
    ans.push_back(StateToIValue({attn, conv}));
  }

  return ans;
}

torch::IValue OnlineConformerTransducerModel::GetEncoderInitStates(
    int32_t /*unused=1*/) {
  torch::NoGradGuard no_grad;
  return encoder_.run_method("get_init_state", left_context_, device_);
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
OnlineConformerTransducerModel::RunEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor &num_processed_frames, torch::IValue states) {
  torch::NoGradGuard no_grad;

  auto outputs =
      encoder_
          .run_method("streaming_forward", features, features_length, states,
                      num_processed_frames, left_context_, right_context_)
          .toTuple();

  torch::IValue encoder_out = outputs->elements()[0];
  auto encoder_out_length = outputs->elements()[1].toTensor();

  auto next_states = outputs->elements()[2];

  auto projected_encoder_out =
      encoder_proj_.run_method("forward", encoder_out).toTensor();

  return std::make_tuple(projected_encoder_out, encoder_out_length,
                         next_states);
}

torch::Tensor OnlineConformerTransducerModel::RunDecoder(
    const torch::Tensor &decoder_input) {
  torch::NoGradGuard no_grad;
  auto decoder_out =
      decoder_.run_method("forward", decoder_input, /*need_pad*/ false);

  return decoder_proj_.run_method("forward", decoder_out).toTensor();
}

torch::Tensor OnlineConformerTransducerModel::RunJoiner(
    const torch::Tensor &encoder_out, const torch::Tensor &decoder_out) {
  torch::NoGradGuard no_grad;
  return joiner_
      .run_method("forward", encoder_out, decoder_out,
                  /*project_input*/ false)
      .toTensor();
}

}  // namespace sherpa
