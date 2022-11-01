/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Wei Kang)
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
#include "sherpa/csrc/ctc_conformer_model.h"

#include <string>
#include <vector>

namespace sherpa {

CtcConformerModel::CtcConformerModel(const std::string &filename,
                                     torch::Device device /*= torch::kCPU*/,
                                     bool optimize_for_inference /*= false*/)
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
}

torch::IValue CtcConformerModel::Forward(
    const std::vector<torch::IValue> &input) {
  return model_(input);
}

torch::Tensor CtcConformerModel::GetLogSoftmaxOut(
    torch::IValue forward_out) const {
  return forward_out.toTuple()->elements()[0].toTensor();
}

torch::Tensor CtcConformerModel::GetLogSoftmaxOutLength(
    torch::IValue forward_out) const {
  return forward_out.toTuple()->elements()[1].toTensor();
}

}  // namespace sherpa
