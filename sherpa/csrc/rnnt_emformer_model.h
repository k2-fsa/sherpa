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
#ifndef SHERPA_CSRC_RNNT_EMFORMER_MODEL_H_
#define SHERPA_CSRC_RNNT_EMFORMER_MODEL_H_

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa/csrc/rnnt_model.h"
#include "torch/script.h"

namespace sherpa {

/** It wraps a torch script model, which is from
 * pruned_stateless_emformer_rnnt2/model.py within icefall.
 */
class RnntEmformerModel : public RnntModel {
 public:
  // for testing in ./test_stack_unstack_states.cc
  RnntEmformerModel() = default;
  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   * @param optimize_for_inference true to invoke
   *                               torch::jit::optimize_for_inference().
   */
  explicit RnntEmformerModel(const std::string &filename,
                             torch::Device device = torch::kCPU,
                             bool optimize_for_inference = false);

  ~RnntEmformerModel() override = default;

  // state[i] contains state for the i-th layer.
  // state[i][k] is either a 3-d tensor of shape (T, N, C) or
  // a 2-d tensor of shape (C, N)
  using State = std::vector<std::vector<torch::Tensor>>;

  torch::IValue GetEncoderInitStates(int32_t unused = 1) override;
  torch::IValue StateToIValue(const State &s) const;
  State StateFromIValue(torch::IValue ivalue) const;

  torch::IValue StackStates(
      const std::vector<torch::IValue> &states) const override;

  std::vector<torch::IValue> UnStackStates(torch::IValue states) const override;

  std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
  StreamingForwardEncoder(const torch::Tensor &features,
                          const torch::Tensor &features_length,
                          const torch::Tensor &unused_num_processed_frames,
                          torch::IValue states) override;

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

  // Hard code the subsampling_factor to 4 here since the subsampling
  // method uses ((len - 1) // 2 - 1) // 2)
  int32_t SubsamplingFactor() const override { return subsampling_factor_; }
  torch::Device Device() const override { return device_; }
  int32_t BlankId() const override { return blank_id_; }
  int32_t UnkId() const override { return unk_id_; }
  int32_t ContextSize() const override { return context_size_; }
  int32_t VocabSize() const override { return vocab_size_; }
  int32_t ChunkLength() const override { return chunk_length_; }
  int32_t PadLength() const override { return pad_length_; }

 private:
  torch::jit::Module model_;

  // The following modules are just aliases to modules in model_
  torch::jit::Module encoder_;
  torch::jit::Module decoder_;
  torch::jit::Module joiner_;

  torch::Device device_{"cpu"};
  int32_t blank_id_;
  int32_t unk_id_;
  int32_t vocab_size_;
  int32_t context_size_;
  int32_t chunk_length_;
  int32_t pad_length_;
  int32_t subsampling_factor_ = 4;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_RNNT_EMFORMER_MODEL_H_
