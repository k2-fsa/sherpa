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

torch::IValue RnntConvEmformerModel::StackStates(
    const std::vector<torch::IValue> &states) const {
  int32_t batch_size = states.size();

  // attn_caches.size() == num_layers
  std::vector<std::vector<std::vector<torch::Tensor>>> attn_caches;
  // We will call torch.stack(attn_caches[i][j]) later

  // conv_caches.size() == num_layers
  std::vector<std::vector<torch::Tensor>> conv_caches;
  // we will call torch.stack(conv_caches[i]) later
  int32_t num_layers = 0;

  for (auto &s : states) {
    // s is a Tuple
    // s[0] contains attn_caches : List[List[torch.Tensor]]
    // s[1] contains conv_caches: List[torch.Tensor]
    //
    // len(attn_caches) == num_layers == len(conv_caches)
    //
    // len(attn_caches[i]) == 3
    // attn_caches[i][0] is a 2-D tensor of shape [memory_size, d_mode]
    // attn_caches[i][1] and attn_caches[i][2] are 2-D tensors of shape
    // [context_size, d_mode]
    auto tuple_ptr = s.toTuple();
    torch::List<torch::IValue> list_attn = tuple_ptr->elements()[0].toList();
    torch::List<torch::IValue> list_conv = tuple_ptr->elements()[1].toList();

    // attn.size() == num_layers
    torch::List<torch::List<torch::Tensor>> attn =
        c10::impl::toTypedList<torch::List<torch::Tensor>>(list_attn);

    torch::List<torch::Tensor> conv =
        c10::impl::toTypedList<torch::Tensor>(list_conv);

    num_layers = attn.size();

    if (attn_caches.empty()) {
      attn_caches.resize(num_layers);
      conv_caches.resize(num_layers);
    }

    for (int32_t l = 0; l != num_layers; ++l) {
      const torch::List<torch::Tensor> &attn_l = attn[l];
      int32_t num_states_this_layer = attn_l.size();

      auto &attn_caches_l = attn_caches[l];
      if (attn_caches_l.empty()) {
        attn_caches_l.resize(num_states_this_layer);
      }

      for (int32_t k = 0; k != num_states_this_layer; ++k) {
        attn_caches_l[k].push_back(attn_l[k]);
      }

      conv_caches[l].push_back(conv[l]);
    }  // for (int32_t l = 0; l != num_layers; ++l)
  }    // for (auto &s : states)

  std::vector<std::vector<torch::Tensor>> stacked_attn_caches(num_layers);
  std::vector<torch::Tensor> stacked_conv_caches(num_layers);

  for (int32_t l = 0; l != num_layers; ++l) {
    auto &attn_caches_l = attn_caches[l];
    auto &stacked_attn_caches_l = stacked_attn_caches[l];
    for (size_t i = 0; i != attn_caches_l.size(); ++i) {
      stacked_attn_caches_l.push_back(
          torch::stack(attn_caches_l[i], /*dim*/ 1));
    }

    stacked_conv_caches[l] = torch::stack(conv_caches[l], /*dim*/ 0);
  }

  return torch::ivalue::Tuple::create(stacked_attn_caches, stacked_conv_caches);
}

std::vector<torch::IValue> RnntConvEmformerModel::UnStackStates(
    torch::IValue states) const {
  TORCH_CHECK(states.isTuple(), "Expect a tuple. Given ", states.tagKind());

  auto tuple_ptr = states.toTuple();
  torch::List<torch::IValue> list_attn = tuple_ptr->elements()[0].toList();
  torch::List<torch::IValue> list_conv = tuple_ptr->elements()[1].toList();

  torch::List<torch::List<torch::Tensor>> stacked_attn =
      c10::impl::toTypedList<torch::List<torch::Tensor>>(list_attn);

  torch::List<torch::Tensor> stacked_conv =
      c10::impl::toTypedList<torch::Tensor>(list_conv);

  int32_t batch_size =
      static_cast<const torch::Tensor &>(stacked_conv[0]).size(0);
  int32_t num_layers = stacked_conv.size();
  int32_t num_states_per_layer =
      static_cast<const torch::List<torch::Tensor> &>(stacked_attn[0]).size();

  std::vector<std::vector<std::vector<torch::Tensor>>> unstacked_attn(
      batch_size);

  for (auto &v : unstacked_attn) {
    v.resize(num_layers);
  }

  std::vector<std::vector<torch::Tensor>> unstacked_conv(batch_size);

  for (int32_t l = 0; l != num_layers; ++l) {
    const torch::List<torch::Tensor> &stacked_attn_l = stacked_attn[l];
    std::vector<std::vector<torch::Tensor>> layer_states(num_states_per_layer);
    for (int32_t k = 0; k != num_states_per_layer; ++k) {
      std::vector<torch::Tensor> s =
          torch::unbind(stacked_attn_l[k], /*dim*/ 1);
      for (int32_t b = 0; b != batch_size; ++b) {
        unstacked_attn[b][l].push_back(std::move(s[b]));
      }
    }  // for (int32_t k = 0; k != num_states_per_layer; ++k)

    auto v = torch::unbind(stacked_conv[l], /*dim*/ 0);
    for (int32_t b = 0; b != batch_size; ++b) {
      unstacked_conv[b].push_back(v[b]);
    }
  }  // for (int32_t l = 0; l != num_layers; ++l)

  std::vector<torch::IValue> ans(batch_size);
  for (int32_t b = 0; b != batch_size; ++b) {
    ans[b] = torch::ivalue::Tuple::create(unstacked_attn[b], unstacked_conv[b]);
  }

  return ans;
}

std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
RnntConvEmformerModel::StreamingForwardEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor &num_processed_frames, torch::IValue states) {
  torch::NoGradGuard no_grad;
  torch::IValue ivalue = encoder_.run_method("infer", features, features_length,
                                             num_processed_frames, states);
  auto tuple_ptr = ivalue.toTuple();
  torch::Tensor encoder_out = tuple_ptr->elements()[0].toTensor();

  torch::Tensor encoder_out_length = tuple_ptr->elements()[1].toTensor();
  torch::IValue next_states = tuple_ptr->elements()[2];

  return std::make_tuple(encoder_out, encoder_out_length, next_states);
}

torch::IValue RnntConvEmformerModel::GetEncoderInitStates(
    int32_t /*batch_size = 1*/) {
  torch::NoGradGuard no_grad;
  return encoder_.run_method("init_states", device_);
}

torch::Tensor RnntConvEmformerModel::ForwardDecoder(
    const torch::Tensor &decoder_input) {
  torch::NoGradGuard no_grad;
  return decoder_.run_method("forward", decoder_input, /*need_pad*/ false)
      .toTensor();
}

torch::Tensor RnntConvEmformerModel::ForwardJoiner(
    const torch::Tensor &projected_encoder_out,
    const torch::Tensor &projected_decoder_out) {
  torch::NoGradGuard no_grad;
  return joiner_
      .run_method("forward", projected_encoder_out, projected_decoder_out,
                  /*project_input*/ false)
      .toTensor();
}

torch::Tensor RnntConvEmformerModel::ForwardEncoderProj(
    const torch::Tensor &encoder_out) {
  torch::NoGradGuard no_grad;
  return encoder_proj_.run_method("forward", encoder_out).toTensor();
}

torch::Tensor RnntConvEmformerModel::ForwardDecoderProj(
    const torch::Tensor &decoder_out) {
  torch::NoGradGuard no_grad;
  return decoder_proj_.run_method("forward", decoder_out).toTensor();
}
}  // namespace sherpa
