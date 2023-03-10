// sherpa/csrc/offline-nemo-enc-dec-ctc-model-bpe.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa/csrc/offline-nemo-enc-dec-ctc-model-bpe.h"

namespace sherpa {

OfflineNeMoEncDecCTCModelBPE::OfflineNeMoEncDecCTCModelBPE(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();

  // TODO(fangjun): This is true for the following models:
  //
  // Citrinet-512:
  // https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_512
  //
  // We need to find a way to embed it into the model
  subsampling_factor_ = 8;
}

torch::IValue OfflineNeMoEncDecCTCModelBPE::Forward(
    torch::Tensor features, torch::Tensor features_length) {
  torch::NoGradGuard no_grad;

  // Change (N, T, C) to (N, C, T)
  features = features.permute({0, 2, 1});

  return model_.run_method("forward", features.to(device_),
                           features_length.to(device_));
}

torch::Tensor OfflineNeMoEncDecCTCModelBPE::GetLogSoftmaxOut(
    torch::IValue forward_out) const {
  auto logit = forward_out.toTensor();
  return logit.roll(1 /*shift right with 1 column*/, 2 /*dim*/);
}

torch::Tensor OfflineNeMoEncDecCTCModelBPE::GetLogSoftmaxOutLength(
    torch::IValue forward_out) const {
  // We return an undefined tensor and the caller should use
  // the features_length and subsampling_factor_ to figure out
  // the actual length
  return {};
}

}  // namespace sherpa
