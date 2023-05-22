// sherpa/csrc/online-zipformer2-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa/csrc/online-zipformer2-transducer-model.h"

#include <array>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace sherpa {

OnlineZipformer2TransducerModel::OnlineZipformer2TransducerModel(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();

  encoder_ = model_.attr("encoder").toModule();
  decoder_ = model_.attr("decoder").toModule();
  joiner_ = model_.attr("joiner").toModule();

  context_size_ =
      decoder_.attr("conv").toModule().attr("weight").toTensor().size(2);

  int32_t pad_length = encoder_.attr("pad_length").toInt();

  chunk_shift_ = encoder_.attr("chunk_size").toInt() * 2;
  chunk_size_ = chunk_shift_ + pad_length;
}

torch::IValue OnlineZipformer2TransducerModel::StackStates(
    const std::vector<torch::IValue> &_states) const {
  torch::NoGradGuard no_grad;

  std::vector<torch::List<torch::Tensor>> states;
  states.reserve(_states.size());
  for (const auto &s : _states) {
    states.push_back(c10::impl::toTypedList<torch::Tensor>(s.toList()));
  }

  std::vector<torch::Tensor> stacked_states;
  stacked_states.reserve(states[0].size());

  int32_t batch_size = static_cast<int32_t>(states.size());
  int32_t num_layers = (static_cast<int32_t>(states[0].size()) - 2) / 6;

  std::vector<torch::Tensor> buf(batch_size);

  std::array<int32_t, 6> batch_dim = {1, 1, 1, 1, 0, 0};

  for (int32_t i = 0; i != num_layers; ++i) {
    // each layer has 6 states
    int32_t offset = i * 6;

    for (int32_t s = 0; s != 6; ++s) {
      for (int32_t b = 0; b != batch_size; ++b) {
        buf[b] = states[b][offset + s];
      }

      stacked_states.push_back(torch::cat(buf, /*dim*/ batch_dim[s]));
    }
  }

  // for the last two tensors
  std::vector<torch::Tensor> buf1(batch_size);
  for (int32_t b = 0; b != batch_size; ++b) {
    buf[b] = states[b][states[0].size() - 2];
    buf1[b] = states[b][states[0].size() - 1];
  }

  stacked_states.push_back(torch::cat(buf, /*dim*/ 0));
  stacked_states.push_back(torch::cat(buf1, /*dim*/ 0));

  return stacked_states;
}

std::vector<torch::IValue> OnlineZipformer2TransducerModel::UnStackStates(
    torch::IValue ivalue) const {
  torch::NoGradGuard no_grad;
  // ivalue is a list
  auto list_ptr = ivalue.toList();
  int32_t num_elements = list_ptr.size();

  std::vector<torch::Tensor> states;
  states.reserve(num_elements);
  for (int32_t i = 0; i != num_elements; ++i) {
    states.emplace_back(list_ptr.get(i).toTensor());
  }

  int32_t num_layers = (states.size() - 2) / 6;
  int32_t batch_size = states[0].size(1);

  std::vector<std::vector<torch::Tensor>> unstacked_states(batch_size);
  for (auto &s : unstacked_states) {
    s.reserve(states.size());
  }

  std::array<int32_t, 6> batch_dim = {1, 1, 1, 1, 0, 0};

  for (int32_t i = 0; i != num_layers; ++i) {
    int32_t offset = 6 * i;

    for (int32_t s = 0; s != 6; ++s) {
      std::vector<torch::Tensor> ss = torch::chunk(
          states[offset + s], /*chunks*/ batch_size, /*dim*/ batch_dim[s]);

      for (int32_t b = 0; b != batch_size; ++b) {
        unstacked_states[b].push_back(std::move(ss[b]));
      }
    }
  }

  // for the last two tensors
  auto ss =
      torch::chunk(states[states.size() - 2], /*chunk*/ batch_size, /*dim*/ 0);
  for (int32_t b = 0; b != batch_size; ++b) {
    unstacked_states[b].push_back(std::move(ss[b]));
  }

  ss = torch::chunk(states[states.size() - 1], /*chunk*/ batch_size, /*dim*/ 0);
  for (int32_t b = 0; b != batch_size; ++b) {
    unstacked_states[b].push_back(std::move(ss[b]));
  }

  std::vector<torch::IValue> ans(batch_size);
  for (int32_t n = 0; n != batch_size; ++n) {
    // unstacked_states[n] is std::vector<torch::Tensor>
    ans[n] = std::move(unstacked_states[n]);
  }

  return ans;
}

torch::IValue OnlineZipformer2TransducerModel::GetEncoderInitStates(
    int32_t batch_size /*=1*/) {
  torch::NoGradGuard no_grad;
  auto states = encoder_.run_method("get_init_states", batch_size, device_);
  /* states is a list of tensors. States of all layers are concatednated into
     a single list.
     State of each layer has 6 tensors:

       - s0: (x, batch_size, x)
       - s1: (1, batch_size, x, x)
       - s2: (x, batch_size, x)
       - s3: (x, batch_size, x)
       - s4: (batch_size, x, x)
       - s5: (batch_size, x, x)


    In addition,
       - states[-1}, a 4-D tensor of shape (batch_size, x, x, x)
       - states[-2], a 1-D tensor of shape (batch_size, )

  If you are curious about the format of the states, please have a look at
    -
  https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/export.py#L363
  https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/streaming_decode.py#L220
   */
  return states;
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
OnlineZipformer2TransducerModel::RunEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor &num_processed_frames, torch::IValue states) {
  torch::NoGradGuard no_grad;

  torch::List<torch::Tensor> s_list =
      c10::impl::toTypedList<torch::Tensor>(states.toList());
  torch::IValue ivalue =
      encoder_.run_method("forward", features, features_length, states);

  auto tuple_ptr = ivalue.toTuple();
  torch::Tensor encoder_out = tuple_ptr->elements()[0].toTensor();

  torch::Tensor encoder_out_length = tuple_ptr->elements()[1].toTensor();

  auto next_states = tuple_ptr->elements()[2];

  return std::make_tuple(encoder_out, encoder_out_length, next_states);
}

torch::Tensor OnlineZipformer2TransducerModel::RunDecoder(
    const torch::Tensor &decoder_input) {
  torch::NoGradGuard no_grad;
  return decoder_
      .run_method("forward", decoder_input,
                  /*need_pad*/ false)
      .toTensor();
}

torch::Tensor OnlineZipformer2TransducerModel::RunJoiner(
    const torch::Tensor &encoder_out, const torch::Tensor &decoder_out) {
  torch::NoGradGuard no_grad;
  return joiner_
      .run_method("forward", encoder_out, decoder_out, /*project_input*/ true)
      .toTensor();
}

}  // namespace sherpa
