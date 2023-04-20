/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang,
 *                                                   Zengwei Yao)
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
#ifndef SHERPA_CSRC_RNNT_LSTM_MODEL_H_
#define SHERPA_CSRC_RNNT_LSTM_MODEL_H_

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa/csrc/rnnt_model.h"
#include "torch/script.h"

namespace sherpa {

/** It wraps a torch script model, which is from
 * lstm_transducer_stateless/model.py within icefall.
 */
class RnntLstmModel : public RnntModel {
 public:
  RnntLstmModel() = default;  // for testing in ./test_stack_unstack_states.cc
  /**
   * @param encoder_filename Path name of the torch script encoder module.
   * @param decoder_filename Path name of the torch script decoder module.
   * @param joiner_filename Path name of the torch script joiner module.
   * @param device  The model will be moved to this device
   * @param optimize_for_inference true to invoke
   *                               torch::jit::optimize_for_inference().
   */
  explicit RnntLstmModel(const std::string &encoder_filename,
                         const std::string &decoder_filename,
                         const std::string &joiner_filename,
                         torch::Device device = torch::kCPU,
                         bool optimize_for_inference = false);

  ~RnntLstmModel() override = default;

  using State = std::pair<torch::Tensor, torch::Tensor>;

  torch::IValue StateToIValue(const State &s) const;
  State StateFromIValue(torch::IValue ivalue) const;

  std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
  StreamingForwardEncoder(const torch::Tensor &features,
                          const torch::Tensor &features_length,
                          const torch::Tensor &unused_num_processed_frames,
                          torch::IValue states) override;

  // The returned IValue is a tuple containing two tensors:
  //  - hx: (num_layers, batch_size, proj_size)
  //  - cx: (num_layers, batch_size, hidden_size)
  //  See icefall/egs/librispeech/ASR/lstm_transducer_stateless2/lstm.py
  //  for details
  torch::IValue GetEncoderInitStates(int32_t batch_size = 1) override;

  torch::IValue StackStates(
      const std::vector<torch::IValue> &states) const override;

  std::vector<torch::IValue> UnStackStates(torch::IValue states) const override;

  /** Run the decoder network.
   *
   * @param decoder_input  A 2-D tensor of shape (N, U).
   * @return Return a tensor of shape (N, U, decoder_dim)
   */
  torch::Tensor ForwardDecoder(const torch::Tensor &decoder_input) override;

  /** Run the joiner network.
   *
   * @param encoder_out  A 2-D tensor of shape (N, C).
   * @param decoder_out  A 2-D tensor of shape (N, C).
   * @return Return a tensor of shape (N, vocab_size)
   */
  torch::Tensor ForwardJoiner(const torch::Tensor &encoder_out,
                              const torch::Tensor &decoder_out) override;

  torch::Device Device() const override { return device_; }
  int32_t BlankId() const override { return blank_id_; }
  int32_t UnkId() const override { return unk_id_; }
  int32_t ContextSize() const override { return context_size_; }
  int32_t VocabSize() const override { return vocab_size_; }
  int32_t ChunkLength() const override { return chunk_length_; }
  int32_t PadLength() const override { return pad_length_; }

 private:
  torch::jit::Module model_;

  // the following modules are just aliases to modules in model_
  torch::jit::Module encoder_;
  torch::jit::Module decoder_;
  torch::jit::Module joiner_;

  torch::Device device_{"cpu"};
  int32_t blank_id_;
  int32_t vocab_size_;
  int32_t unk_id_;
  int32_t context_size_;
  int32_t chunk_length_;
  int32_t pad_length_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_RNNT_LSTM_MODEL_H_
