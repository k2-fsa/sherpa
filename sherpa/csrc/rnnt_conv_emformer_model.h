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
#ifndef SHERPA_CSRC_RNNT_CONV_EMFORMER_MODEL_H_
#define SHERPA_CSRC_RNNT_CONV_EMFORMER_MODEL_H_

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
class RnntConvEmformerModel : public RnntModel {
 public:
  /**
   * @param filename Path name of the torch script model.
   * @param device  The model will be moved to this device
   * @param optimize_for_inference true to invoke
   *                               torch::jit::optimize_for_inference().
   */
  explicit RnntConvEmformerModel(const std::string &filename,
                                 torch::Device device = torch::kCPU,
                                 bool optimize_for_inference = false);

  ~RnntConvEmformerModel() override = default;

  using State = std::pair<std::vector<std::vector<torch::Tensor>>,
                          std::vector<torch::Tensor>>;

  std::tuple<torch::Tensor, torch::Tensor, State> StreamingForwardEncoder(
      const torch::Tensor &features, const torch::Tensor &features_length,
      const torch::Tensor &num_processed_frames, State states);

  State GetEncoderInitStates();

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

  torch::Device Device() const override { return device_; }
  int32_t BlankId() const override { return blank_id_; }
  int32_t UnkId() const override { return unk_id_; }
  int32_t ContextSize() const override { return context_size_; }
  int32_t VocabSize() const override { return vocab_size_; }
  int32_t ChunkLength() const { return chunk_length_; }
  int32_t RightContextLength() const { return right_context_length_; }
  int32_t PadLength() const { return pad_length_; }

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
  int32_t vocab_size_;
  int32_t unk_id_;
  int32_t context_size_;
  int32_t chunk_length_;
  int32_t right_context_length_;
  int32_t pad_length_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_RNNT_CONV_EMFORMER_MODEL_H_
