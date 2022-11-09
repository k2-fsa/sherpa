// sherpa/csrc/ctc-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation
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
