// sherpa/csrc/online-transducer-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_TRANSDUCER_MODEL_H_
#define SHERPA_CSRC_ONLINE_TRANSDUCER_MODEL_H_

#include <tuple>
#include <vector>

#include "torch/script.h"

namespace sherpa {

class OnlineTransducerModel {
 public:
  virtual ~OnlineTransducerModel() = default;

  /** Stack a list of individual states into a batch.
   *
   * It is the inverse operation of `UnStackStates`.
   *
   * @param states states[i] contains the state for the i-th utterance.
   * @return Return a single value representing the batched state.
   */
  virtual torch::IValue StackStates(
      const std::vector<torch::IValue> &states) const = 0;

  /** Unstack a batch state into a list of individual states.
   *
   * It is the inverse operation of `StackStates`.
   *
   * @param states A batched state.
   * @return ans[i] contains the state for the i-th utterance.
   */
  virtual std::vector<torch::IValue> UnStackStates(
      torch::IValue states) const = 0;

  /** Get the initial encoder states.
   *
   * @param unused A placeholder. Some models, e.g., ConvEmformer uses it, will
   *               other models won't use it.
   * @return Return the initial encoder state.
   */
  virtual torch::IValue GetEncoderInitStates(int32_t unused = 1) = 0;

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
  std::tuple<torch::Tensor, torch::Tensor, torch::IValue> virtual RunEncoder(
      const torch::Tensor &features, const torch::Tensor &features_length,
      const torch::Tensor &num_processed_frames, torch::IValue states) = 0;

  /** Run the decoder network.
   *
   * Caution: We assume there are no recurrent connections in the decoder and
   *          the decoder is stateless. See
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/decoder.py
   *          for an example
   *
   * @param decoder_input It is usually of shape (N, context_size)
   * @return Return a tensor of shape (N, 1, decoder_dim).
   */
  virtual torch::Tensor RunDecoder(const torch::Tensor &decoder_input) = 0;

  /** Run the joint network.
   *
   * @param encoder_out Output of the encoder network. A tensor of shape
   *                    (N, encoder_dim).
   * @param decoder_out Output of the decoder network. A tensor of shape
   *                    (N, decoder_dim).
   * @return Return a tensor of shape (N, vocab_size). In icefall, the last
   *         last layer of the joint network is `nn.Linear`,
   *         not `nn.LogSoftmax`.
   */
  virtual torch::Tensor RunJoiner(const torch::Tensor &encoder_out,
                                  const torch::Tensor &decoder_out) = 0;

  /** Return the device where computation takes place.
   *
   * Note: We don't support moving the model to a different device
   *       after construction.
   */
  virtual torch::Device Device() const = 0;

  /** If we are using a stateless decoder and if it contains a
   *  Conv1D, this function returns the kernel size of the convolution layer.
   */
  virtual int32_t ContextSize() const { return 0; }

  /** We send this number of feature frames to the encoder at a time. */
  virtual int32_t ChunkSize() const = 0;

  /** Number of input frames to discard after each call to RunEncoder.
   *
   * For instance, if we have 30 frames, chunk_size=8, chunk_shift=6.
   *
   * In the first call of RunEncoder, we use frames 0~7 since chunk_size is 8.
   * Then we discard frame 0~5 since chunk_shift is 6.
   * In the second call of RunEncoder, we use frames 6~13; and then we discard
   * frames 6~11.
   * In the third call of RunEncoder, we use frames 12~19; and then we discard
   * frames 12~16.
   */
  virtual int32_t ChunkShift() const = 0;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_TRANSDUCER_MODEL_H_
