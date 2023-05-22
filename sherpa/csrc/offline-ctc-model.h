// sherpa/csrc/offline-ctc-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_CTC_MODEL_H_
#define SHERPA_CSRC_OFFLINE_CTC_MODEL_H_

#include <vector>

#include "torch/script.h"

namespace sherpa {

class OfflineCtcModel {
 public:
  virtual ~OfflineCtcModel() = default;

  // Subsampling factor of the model
  virtual int32_t SubsamplingFactor() const = 0;

  // Number of modeling unit. Should be equal to
  // GetLogSoftmaxOut().size(-1)
  int32_t VocabSize() const { return vocab_size_; }

  // Return the underlying device where computation would happen
  virtual torch::Device Device() const = 0;

  /** Run the model with a given input.
   *
   * @param features  A 3-D tensor of shape (N, T, C).
   * @param features_length A 1-D tensor of shape (N,).
   */
  virtual torch::IValue Forward(torch::Tensor features,
                                torch::Tensor features_length) = 0;

  // Get the log softmax output of the network from the output of Forward
  // method.
  // The returned tensor has shape (N, T, C).
  virtual torch::Tensor GetLogSoftmaxOut(torch::IValue forward_out) const = 0;

  // Get the output length before padding from the output of Forward method.
  // The returned tensor has shape (N,)
  virtual torch::Tensor GetLogSoftmaxOutLength(
      torch::IValue forward_out) const = 0;

  // Send some fake data to the model for computation
  virtual void WarmUp(torch::Tensor features, torch::Tensor features_length) {
    auto ivalue = Forward(features, features_length);
    auto log_prob = GetLogSoftmaxOut(ivalue);
    vocab_size_ = log_prob.size(-1);
  }

 protected:
  int32_t vocab_size_ = -1;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_CTC_MODEL_H_
