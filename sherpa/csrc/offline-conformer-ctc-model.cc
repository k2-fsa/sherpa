// sherpa/csrc/offline-conformer-ctc-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/csrc/offline-conformer-ctc-model.h"

#include <string>
#include <vector>

namespace sherpa {

OfflineConformerCtcModel::OfflineConformerCtcModel(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();
}

torch::IValue OfflineConformerCtcModel::Forward(torch::Tensor features,
                                                torch::Tensor features_length) {
  torch::NoGradGuard no_grad;

  int32_t batch_size = features.size(0);

  torch::Dict<std::string, torch::Tensor> sup;
  sup.insert("sequence_idx", torch::arange(batch_size, torch::kInt));
  sup.insert("start_frame", torch::zeros({batch_size}, torch::kInt));
  sup.insert("num_frames", features_length.cpu().to(torch::kInt));

  torch::IValue supervisions(sup);

  return model_.run_method("forward", features.to(device_), sup);
}

torch::Tensor OfflineConformerCtcModel::GetLogSoftmaxOut(
    torch::IValue forward_out) const {
  return forward_out.toTuple()->elements()[0].toTensor();
}

torch::Tensor OfflineConformerCtcModel::GetLogSoftmaxOutLength(
    torch::IValue forward_out) const {
  torch::NoGradGuard no_grad;

  auto mask = forward_out.toTuple()->elements()[2].toTensor();
  return (~mask).sum(1);
}

}  // namespace sherpa
