// sherpa/csrc/offline-nemo-enc-dec-ctc-model-bpe.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_BPE_H_
#define SHERPA_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_BPE_H_

#include <string>
#include <vector>

#include "sherpa/csrc/offline-ctc-model.h"
namespace sherpa {

/** This class models the EncDecCTCModelBPE model from NeMo.
 *
 * See
 * https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_bpe_models.py
 * https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py
 */
class OfflineNeMoEncDecCTCModelBPE : public OfflineCtcModel {
 public:
  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   */
  explicit OfflineNeMoEncDecCTCModelBPE(const std::string &filename,
                                        torch::Device device = torch::kCPU);

  torch::Device Device() const override { return device_; }

  int32_t SubsamplingFactor() const override { return subsampling_factor_; }

  /** Run the encoder of the model.
   *
   * See
   * https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py#L196
   * for its documentation in Python.
   *
   * @param features A 3-D tensor of shape (N, T, C).
   *                 Caution: We permute it to (N, C, T) inside.
   * @param features_length A 3-D tensor of shape (N,)
   * @return Return a 3-D tensor of shape (N, T, C). It represents
   *         the log_prob.
   */
  torch::IValue Forward(torch::Tensor features,
                        torch::Tensor features_length) override;

  /** Note: In NeMo, the last column of forward_out represent blank.
   * We move it to the first column in this function.
   */
  torch::Tensor GetLogSoftmaxOut(torch::IValue forward_out) const override;

  torch::Tensor GetLogSoftmaxOutLength(
      torch::IValue forward_out) const override;

  // we need to set the subsampling_factor_ inside it
  void WarmUp(torch::Tensor features, torch::Tensor features_length) override;

 private:
  torch::Device device_;
  torch::jit::Module model_;
  int32_t subsampling_factor_ = 0;
};

using OfflineNeMoEncDecCTCModel = OfflineNeMoEncDecCTCModelBPE;

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_NEMO_ENC_DEC_CTC_MODEL_BPE_H_
