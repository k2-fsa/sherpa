// sherpa/csrc/offline-conformer-transducer-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/offline-conformer-transducer-model.h"

#include <string>
#include <utility>

namespace sherpa {

OfflineConformerTransducerModel::OfflineConformerTransducerModel(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  // See
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/model.py#L29
  // for the definition of `model_`.
  //
  // Note: pruned_transducer_statelessX where X>=2 has the same model
  // architecture. We use pruned_transducer_stateless2 as an exmaple here, but
  // it applies also to pruned_transducer_stateless3,
  // pruned_transducer_stateless4, etc.
  model_ = torch::jit::load(filename, device);
  model_.eval();

  encoder_ = model_.attr("encoder").toModule();
  decoder_ = model_.attr("decoder").toModule();
  joiner_ = model_.attr("joiner").toModule();

  encoder_proj_ = joiner_.attr("encoder_proj").toModule();
  decoder_proj_ = joiner_.attr("decoder_proj").toModule();

  context_size_ = decoder_.attr("context_size").toInt();
}

std::pair<torch::Tensor, torch::Tensor>
OfflineConformerTransducerModel::RunEncoder(
    const torch::Tensor &features, const torch::Tensor &features_length) {
  torch::NoGradGuard no_grad;

  auto outputs =
      encoder_.run_method("forward", features, features_length).toTuple();

  auto encoder_out = outputs->elements()[0];
  auto encoder_out_length = outputs->elements()[1].toTensor();

  auto projected_encoder_out =
      encoder_proj_.run_method("forward", encoder_out).toTensor();

  return {projected_encoder_out, encoder_out_length};
}

torch::Tensor OfflineConformerTransducerModel::RunDecoder(
    const torch::Tensor &decoder_input) {
  torch::NoGradGuard no_grad;
  auto decoder_out =
      decoder_.run_method("forward", decoder_input, /*need_pad*/ false);

  return decoder_proj_.run_method("forward", decoder_out).toTensor();
}

torch::Tensor OfflineConformerTransducerModel::RunJoiner(
    const torch::Tensor &encoder_out, const torch::Tensor &decoder_out) {
  torch::NoGradGuard no_grad;
  return joiner_
      .run_method("forward", encoder_out, decoder_out,
                  /*project_input*/ false)
      .toTensor();
}

}  // namespace sherpa
