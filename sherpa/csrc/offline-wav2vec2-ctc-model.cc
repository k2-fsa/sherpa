// sherpa/csrc/offline-wav2vec2-ctc-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/offline-wav2vec2-ctc-model.h"
namespace sherpa {

OfflineWav2Vec2CtcModel::OfflineWav2Vec2CtcModel(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();
}

torch::IValue OfflineWav2Vec2CtcModel::Forward(torch::Tensor waveforms,
                                               torch::Tensor lengths) {
  torch::NoGradGuard no_grad;

  return model_.run_method("forward", waveforms.to(device_),
                           lengths.to(device_));
}

torch::Tensor OfflineWav2Vec2CtcModel::GetLogSoftmaxOut(
    torch::IValue forward_out) const {
  torch::NoGradGuard no_grad;

  auto logit = forward_out.toTuple()->elements()[0].toTensor();
  return logit.log_softmax(-1);
}

torch::Tensor OfflineWav2Vec2CtcModel::GetLogSoftmaxOutLength(
    torch::IValue forward_out) const {
  torch::NoGradGuard no_grad;

  return forward_out.toTuple()->elements()[1].toTensor();
}

}  // namespace sherpa
