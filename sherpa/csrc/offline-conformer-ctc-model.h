// sherpa/csrc/offline-conformer-ctc-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_CONFORMER_CTC_MODEL_H_
#define SHERPA_CSRC_OFFLINE_CONFORMER_CTC_MODEL_H_

#include <string>
#include <vector>

#include "sherpa/csrc/offline-ctc-model.h"
namespace sherpa {

/** This class models the Conformer model from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/train.py#L668
 */
class OfflineConformerCtcModel : public OfflineCtcModel {
 public:
  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   */
  explicit OfflineConformerCtcModel(const std::string &filename,
                                    torch::Device device = torch::kCPU);

  torch::Device Device() const override { return device_; }

  int32_t SubsamplingFactor() const override { return 4; }

  /** Run the forward method of the model.
   * See
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/transformer.py#L162
   * for its documentation in Python.
   */
  torch::IValue Forward(torch::Tensor features,
                        torch::Tensor features_length) override;

  torch::Tensor GetLogSoftmaxOut(torch::IValue forward_out) const override;

  torch::Tensor GetLogSoftmaxOutLength(
      torch::IValue forward_out) const override;

 private:
  torch::Device device_;
  torch::jit::Module model_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_CONFORMER_CTC_MODEL_H_
