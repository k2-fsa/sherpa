// sherpa/csrc/offline-wenet-conformer-ctc-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/offline-wenet-conformer-ctc-model.h"

namespace sherpa {

OfflineWenetConformerCtcModel::OfflineWenetConformerCtcModel(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();

  subsampling_factor_ = model_.run_method("subsampling_rate").toInt();
}

torch::IValue OfflineWenetConformerCtcModel::Forward(
    torch::Tensor features, torch::Tensor features_length) {
  torch::NoGradGuard no_grad;

  return model_.attr("encoder").toModule().run_method(
      "forward", features.to(device_), features_length.to(device_));
}

torch::Tensor OfflineWenetConformerCtcModel::GetLogSoftmaxOut(
    torch::IValue forward_out) const {
  torch::NoGradGuard no_grad;

  auto logit = forward_out.toTuple()->elements()[0];
  return model_.attr("ctc")
      .toModule()
      .run_method("log_softmax", logit)
      .toTensor();
}

torch::Tensor OfflineWenetConformerCtcModel::GetLogSoftmaxOutLength(
    torch::IValue forward_out) const {
  torch::NoGradGuard no_grad;

  auto mask = forward_out.toTuple()->elements()[1].toTensor();
  return mask.sum({1, 2});
}

}  // namespace sherpa
