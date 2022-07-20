/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang,
 *                                                   Wei kang)
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
#ifndef SHERPA_CSRC_RNNT_CONFORMER_MODEL_H_
#define SHERPA_CSRC_RNNT_CONFORMER_MODEL_H_

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa/csrc/rnnt_model.h"
#include "torch/script.h"

namespace sherpa {

/** It wraps a torch script model, which is from
 * pruned_transducer_stateless2/model.py or
 * pruned_transducer_stateless4/model.py within icefall.
 */
class RnntConformerModel : public RnntModel {
 public:
  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   * @param optimize_for_inference true to invoke
   *                               torch::jit::optimize_for_inference().
   */
  explicit RnntConformerModel(const std::string &filename,
                              torch::Device device = torch::kCPU,
                              bool optimize_for_inference = false);

  ~RnntConformerModel() override = default;

  using State = std::vector<torch::Tensor>;

  State GetEncoderInitStates(int32_t left_context);

  torch::Device Device() const override { return device_; }

  int32_t BlankId() const override { return blank_id_; }
  int32_t UnkId() const override { return unk_id_; }
  int32_t ContextSize() const override { return context_size_; }
  int32_t VocabSize() const override { return vocab_size_; }
  // Hard code the subsampling_factor to 4 here since the subsampling
  // method uses ((len - 1) // 2 - 1) // 2)
  int32_t SubSamplingFactor() const { return 4; }

  /** Run the encoder network.
   *
   * @param features  A 3-D tensor of shape (N, T, C).
   * @param features_length A 1-D tensor of shape (N,) containing the number of
   *                       valid frames in `features`.
   * @return Return a pair containing two tensors:
   *         - encoder_out, a 3-D tensor of shape (N, T, C)
   *         - encoder_out_length, a 1-D tensor of shape (N,) containing the
   *           number of valid frames in `encoder_out`.
   */
  std::pair<torch::Tensor, torch::Tensor> ForwardEncoder(
      const torch::Tensor &features, const torch::Tensor &features_length);

  /** Run the encoder network in streaming mode.
   *
   * @param features  A 3-D tensor of shape (N, T, C).
   * @param features_length A 1-D tensor of shape (N,) containing the number of
   *                       valid frames in `features`.
   * @param states A list of tensors containing the decode caches of previous
   *              frames. It is almost transparent to users, initially this
   *              comes from the return value of `GetEncoderInitStates`, then it
   *              will be updated after finishing each chunk.
   * @param processed_lengths How many frames have processed until now.
   * @param left_context How many previous frames can be seen for current
   *                     chunk.
   * @param right_context How many future frames can be seen for current
   *                      chunk.
   * @return Return a pair containing two tensors:
   *         - encoder_out, a 3-D tensor of shape (N, T, C)
   *         - encoder_out_length, a 1-D tensor of shape (N,) containing the
   *           number of valid frames in `encoder_out`.
   */
  std::tuple<torch::Tensor, torch::Tensor, State> StreamingForwardEncoder(
      const torch::Tensor &features, const torch::Tensor &features_length,
      const State &states, const torch::Tensor &processed_frames,
      int32_t left_context, int32_t right_context);

  /** Run the decoder network.
   *
   * @param decoder_input  A 2-D tensor of shape (N, U).
   * @return Return a tensor of shape (N, U, decoder_dim)
   */
  torch::Tensor ForwardDecoder(const torch::Tensor &decoder_input) override;

  /** Run the joiner network.
   *
   * @param projected_encoder_out  A 3-D tensor of shape (N, T, C).
   * @param projected_decoder_out  A 3-D tensor of shape (N, U, C).
   * @return Return a tensor of shape (N, T, U, vocab_size)
   */
  torch::Tensor ForwardJoiner(
      const torch::Tensor &projected_encoder_out,
      const torch::Tensor &projected_decoder_out) override;

  /** Run the joiner.encoder_proj network.
   *
   * @param encoder_out  The output from the encoder, which is of shape (N,T,C).
   * @return Return a tensor of shape (N, T, joiner_dim).
   */
  torch::Tensor ForwardEncoderProj(const torch::Tensor &encoder_out) override;

  /** Run the joiner.decoder_proj network.
   *
   * @param decoder_out  The output from the encoder, which is of shape (N,T,C).
   * @return Return a tensor of shape (N, T, joiner_dim).
   */
  torch::Tensor ForwardDecoderProj(const torch::Tensor &decoder_out) override;

 private:
  torch::jit::Module model_;

  // The following modules are just aliases to modules in model_
  torch::jit::Module encoder_;
  torch::jit::Module decoder_;
  torch::jit::Module joiner_;
  torch::jit::Module encoder_proj_;
  torch::jit::Module decoder_proj_;

  torch::Device device_;
  int32_t blank_id_;
  int32_t unk_id_;
  int32_t context_size_;
  int32_t vocab_size_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_RNNT_CONFORMER_MODEL_H_
