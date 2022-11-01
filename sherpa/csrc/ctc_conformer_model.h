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
#ifndef SHERPA_CSRC_CTC_CONFORMER_MODEL_H_
#define SHERPA_CSRC_CTC_CONFORMER_MODEL_H_

#include <string>
#include <vector>

#include "sherpa/csrc/ctc_model.h"
namespace sherpa {

/** This class models the Conformer model from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/train.py#L668
 */
class CtcConformerModel : public CtcModel {
 public:
  ~CtcConformerModel() override = default;

  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   * @param optimize_for_inference true to invoke
   *                               torch::jit::optimize_for_inference().
   */
  explicit CtcConformerModel(const std::string &filename,
                             torch::Device device = torch::kCPU,
                             bool optimize_for_inference = false);

  torch::Device Device() const override { return device_; }

  /** Run the forward method of the model.
   * See
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/transformer.py#L162
   * for its documentation in Python.
   *
   * @param input It has two element. The first element contains the 3-D
   *              features of shape (N, T, C); while the second element
   *              contains the supervision_segments. See the above link
   *              for the format of it.
   *
   * @return Return a tuple containing 3 elements, but we only use the first 2
   *         for CTC decoding. The first element contains the log_softmax output
   *         of the model with shape (N, T', C').
   *         The second element contains number of frames of the first
   *         element before padding.
   */
  torch::IValue Forward(const std::vector<torch::IValue> &input) override;

  torch::Tensor GetLogSoftmaxOut(torch::IValue forward_out) const override;

  torch::Tensor GetLogSoftmaxOutLength(
      torch::IValue forward_out) const override;

 private:
  torch::Device device_;
  torch::jit::Module model_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_CTC_CONFORMER_MODEL_H_
