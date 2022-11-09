// sherpa/csrc/offline-wenet-conformer-ctc-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_WENET_CONFORMER_CTC_MODEL_H_
#define SHERPA_CSRC_OFFLINE_WENET_CONFORMER_CTC_MODEL_H_

#include <string>
#include <vector>

#include "sherpa/csrc/offline-ctc-model.h"
namespace sherpa {

/** This class models the Conformer model from wenet.
 *
 * See
 * https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/asr_model.py
 */
class OfflineWenetConformerCtcModel : public OfflineCtcModel {
 public:
  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   * @param optimize_for_inference true to invoke
   *                               torch::jit::optimize_for_inference().
   */
  explicit OfflineWenetConformerCtcModel(const std::string &filename,
                                         torch::Device device = torch::kCPU);

  torch::Device Device() const override { return device_; }

  int32_t SubsamplingFactor() const override { return subsampling_factor_; }

  /** Run the encoder of the model.
   *
   * See
   * https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/asr_model.py#L42
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
  int32_t subsampling_factor_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_WENET_CONFORMER_CTC_MODEL_H_
