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
#ifndef SHERPA_CSRC_CTC_MODEL_H_
#define SHERPA_CSRC_CTC_MODEL_H_

#include <vector>

#include "torch/script.h"

namespace sherpa {

class CtcModel {
 public:
  virtual ~CtcModel() = default;

  // Subsampling factor of the model
  virtual int32_t SubsamplingFactor() const { return 4; }

  // Return the underlying device where computation would happen
  virtual torch::Device Device() const = 0;

  // Run the model with a given input.
  virtual torch::IValue Forward(const std::vector<torch::IValue> &input) = 0;

  // Get the log softmax output of the network from the output of Forward
  // method.
  // The returned tensor has shape (N, T, C).
  virtual torch::Tensor GetLogSoftmaxOut(torch::IValue forward_out) const = 0;

  // Get the output length before padding from the output of Forward method.
  // The returned tensor has shape (N,)
  virtual torch::Tensor GetLogSoftmaxOutLength(
      torch::IValue forward_out) const = 0;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_CTC_MODEL_H_
