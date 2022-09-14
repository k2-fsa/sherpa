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
#ifndef SHERPA_CSRC_RNNT_MODEL_H_
#define SHERPA_CSRC_RNNT_MODEL_H_

#include <string>
#include <utility>

#include "torch/script.h"

namespace sherpa {

/** The base class of stateless transducer model, it has an encoder, decoder
 *  and joiner, and the decoder is stateless.
 *  See the code in pruned_transducer_statelessX/model.py in icefall
 *  repo for for more details.
 */

class RnntModel {
 public:
  virtual ~RnntModel() = default;

  virtual torch::Device Device() const = 0;

  virtual int32_t BlankId() const = 0;
  virtual int32_t UnkId() const = 0;
  virtual int32_t ContextSize() const = 0;
  virtual int32_t VocabSize() const = 0;

  int32_t SubsamplingFactor() const { return 4; }

  /** Run the decoder network.
   *
   * @param decoder_input  A 2-D tensor of shape (N, U).
   * @return Return a tensor of shape (N, U, decoder_dim)
   */
  virtual torch::Tensor ForwardDecoder(const torch::Tensor &decoder_input) = 0;

  /** Run the joiner network.
   *
   * @param projected_encoder_out  A 2-D tensor of shape (N, C).
   * @param projected_decoder_out  A 2-D tensor of shape (N, C).
   * @return Return a tensor of shape (N, vocab_size)
   */
  virtual torch::Tensor ForwardJoiner(
      const torch::Tensor &projected_encoder_out,
      const torch::Tensor &projected_decoder_out) = 0;

  /** Run the joiner.encoder_proj network.
   *
   * @param encoder_out  The output from the encoder, which is of shape (N,T,C).
   * @return Return a tensor of shape (N, T, joiner_dim).
   */
  virtual torch::Tensor ForwardEncoderProj(const torch::Tensor &encoder_out) {
    return encoder_out;
  }

  /** Run the joiner.decoder_proj network.
   *
   * @param decoder_out  The output from the encoder, which is of shape (N,T,C).
   * @return Return a tensor of shape (N, T, joiner_dim).
   */
  virtual torch::Tensor ForwardDecoderProj(const torch::Tensor &decoder_out) {
    return decoder_out;
  }
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_RNNT_MODEL_H_
