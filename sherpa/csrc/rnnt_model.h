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
#include <tuple>
#include <utility>
#include <vector>

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

  virtual int32_t SubsamplingFactor() const { return 4; }

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

  // ------------------------------------------------------------
  // The following methods are for streaming models
  // ------------------------------------------------------------

  // for streaming models
  virtual int32_t ChunkLength() const {
    assert("don't call me" && false);
    exit(-1);
    return 0;
  }

  virtual int32_t PadLength() const {
    assert("don't call me" && false);
    exit(-1);
    return 0;
  }

  /**
   *
   * @param states A list of encoder network states. states[i] is the state
   *               for the i-th stream.
   * @return A state for a batch of streams.
   */
  virtual torch::IValue StackStates(
      const std::vector<torch::IValue> &states) const {
    assert("don't call me" && false);
    exit(-1);
    return {};
  }

  /** Inverse operation of StackStates.
   *
   * @param states  State of the encoder network for a batch of streams.
   *
   * @return A list of encoder network states.
   */
  virtual std::vector<torch::IValue> UnStackStates(torch::IValue states) const {
    assert("don't call me" && false);
    exit(-1);
    return {};
  }

  /** Run the encoder.
   *
   * @param features  A tensor of shape (N, T, C).
   * @param features_length  A tensor of shape (N,) containing the number
   *                         of valid frames in `features` before padding.
   * @param num_processed_frames  Number of processed frames so far before
   *                              subsampling.
   * @param states  Encoder state of the previous chunk.
   *
   * @return Return a tuple containing:
   *           - encoder_out, a tensor of shape (N, T', encoder_out_dim)
   *           - encoder_out_lens, a tensor of shape (N,)
   *           - next_states  Encoder state for the next chunk.
   */
  /* clang-format off */
  std::tuple<torch::Tensor, torch::Tensor, torch::IValue>
  virtual StreamingForwardEncoder(const torch::Tensor &features,
                                  const torch::Tensor &features_length,
                                  const torch::Tensor &num_processed_frames,
                                  torch::IValue states) {
    /* clang-format on*/
    assert("don't call me" && false);
    exit(-1);
    return {};
  }

  virtual torch::IValue GetEncoderInitStates(int32_t batch_size = 1) {
    assert("don't call me" && false);
    exit(-1);
    return {};
  }
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_RNNT_MODEL_H_
