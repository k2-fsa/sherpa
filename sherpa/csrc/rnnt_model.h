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
#ifndef SHERPA_CSRC_RNNT_MODEL_H_
#define SHERPA_CSRC_RNNT_MODEL_H_

#include <tuple>
#include <utility>

#include "torch/script.h"

namespace sherpa {

/** It wraps a torch script model, which is from
 * pruned_transducer_stateless2/mode.py within icefall.
 */
class RnntModel {
 public:
  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   * @param optimize_for_inference true to invoke
   *                               torch::jit::optimize_for_inference().
   */
  explicit RnntModel(const std::string &filename,
                     torch::Device device = torch::kCPU,
                     bool optimize_for_inference = false);

  ~RnntModel() = default;

  torch::Device Device() const { return device_; }

  int32_t BlankId() const { return blank_id_; }
  int32_t UnkId() const { return unk_id_; }
  int32_t ContextSize() const { return context_size_; }

  /** Run the encoder network.
   *
   * @param features  A 3-D tensor of shape (N, T, C).
   * @param features_length A 1-D tensor of shape (N,) containing the number of
   *                       valid frames in `features`.
   * @return Return a tuple containing two tensors:
   *         - encoder_out, a 3-D tensor of shape (N, T, C)
   *         - encoder_out_length, a 1-D tensor of shape (N,) containing the
   *           number of valid frames in `encoder_out`.
   */
  std::pair<torch::Tensor, torch::Tensor> ForwardEncoder(
      const torch::Tensor &features, const torch::Tensor &features_length);

  /** Run the decoder network.
   *
   * @param decoder_input  A 2-D tensor of shape (N, U).
   * @return Return a tensor of shape (N, U, decoder_dim)
   */
  torch::Tensor ForwardDecoder(const torch::Tensor &decoder_input);

  /** Run the joiner network.
   *
   * @param projected_encoder_out  A 3-D tensor of shape (N, T, C).
   * @param projected_decoder_out  A 3-D tensor of shape (N, U, C).
   * @return Return a tensor of shape (N, T, U, vocab_size)
   */
  torch::Tensor ForwardJoiner(const torch::Tensor &projected_encoder_out,
                              const torch::Tensor &projected_decoder_out);

  /** Run the joiner.encoder_proj network.
   *
   * @param encoder_out  The output from the encoder, which is of shape (N,T,C).
   * @return Return a tensor of shape (N, T, joiner_dim).
   */
  torch::Tensor ForwardEncoderProj(const torch::Tensor &encoder_out);

  /** Run the joiner.decoder_proj network.
   *
   * @param decoder_out  The output from the encoder, which is of shape (N,T,C).
   * @return Return a tensor of shape (N, T, joiner_dim).
   */
  torch::Tensor ForwardDecoderProj(const torch::Tensor &decoder_out);

  /** TODO(fangjun): Implement it
   *
   * Run the encoder network in a streaming fashion.
   *
   * @param features  A 3-D tensor of shape (N, T, C).
   * @param features_length  A 1-D tensor of shape (N,) containing the number of
   *                         valid frames in `features`.
   * @param prev_state  It contains the previous state from the encoder network.
   *
   * @return Return a tuple containing 3 entries:
   *          - encoder_out, a 3-D tensor of shape (N, T, C)
   *          - encoder_out_length, a 1-D tensor of shape (N,) containing the
   *            number of valid frames in encoder_out
   *          - next_state, the state for the encoder network.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
  StreamingForwardEncoder(const torch::Tensor &features,
                          const torch::Tensor &feature_lengths,
                          torch::IValue prev_state);

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
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_RNNT_MODEL_H_
