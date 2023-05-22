// sherpa/csrc/offline-wav2vec2-ctc-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_WAV2VEC2_CTC_MODEL_H_
#define SHERPA_CSRC_OFFLINE_WAV2VEC2_CTC_MODEL_H_

#include <string>
#include <vector>

#include "sherpa/csrc/offline-ctc-model.h"
namespace sherpa {

/** This class models the Conformer model from icefall.
 *
 * See
 * https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/model.py#L11
 */
class OfflineWav2Vec2CtcModel : public OfflineCtcModel {
 public:
  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   */
  explicit OfflineWav2Vec2CtcModel(const std::string &filename,
                                   torch::Device device = torch::kCPU);

  torch::Device Device() const override { return device_; }

  int32_t SubsamplingFactor() const override {
    // See Section 4.2 of
    // https://arxiv.org/pdf/2006.11477.pdf
    return 1;
  }

  /** Run the forward method of the model.
   * See
   * https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/model.py#L90
   * for its documentation in Python.
   */
  torch::IValue Forward(torch::Tensor waveforms,
                        torch::Tensor lengths) override;

  torch::Tensor GetLogSoftmaxOut(torch::IValue forward_out) const override;

  torch::Tensor GetLogSoftmaxOutLength(
      torch::IValue forward_out) const override;

 private:
  torch::Device device_;
  torch::jit::Module model_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_WAV2VEC2_CTC_MODEL_H_
