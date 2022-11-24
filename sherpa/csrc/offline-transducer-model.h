// sherpa/csrc/offline-transducer-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_TRANSDUCER_MODEL_H_
#define SHERPA_CSRC_OFFLINE_TRANSDUCER_MODEL_H_

#include <utility>

#include "torch/script.h"

namespace sherpa {

class OfflineTransducerModel {
 public:
  virtual ~OfflineTransducerModel() = default;

  /** Run the encoder network.
   *
   * @param features  A 3-D tensor of shape (N, T, C)
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *
   * @return Return a pair containing:
   *  - encoder_out: A 3-D tensor of shape (N, T', encoder_dim)
   *  - encoder_out_length: A 1-D tensor of shape (N,) containing number
   *                        of frames in `encoder_out` before padding.
   */
  virtual std::pair<torch::Tensor, torch::Tensor> RunEncoder(
      const torch::Tensor &features, const torch::Tensor &features_length) = 0;

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
  virtual int32_t ContextSize() const = 0;

  int32_t VocabSize() const { return vocab_size_; }

  void WarmUp(torch::Tensor features, torch::Tensor features_length) {
    torch::Tensor encoder_out;
    torch::Tensor encoder_out_length;

    std::tie(encoder_out, encoder_out_length) =
        RunEncoder(features, features_length);
    // encoder_out.shape (N, T, joiner_dim)
    // encoder_out_length.shape (N,)

    auto cur_encoder_out = encoder_out.index({torch::indexing::Slice(), 0});
    // cur_encoder_out.shape (N, joiner_dim)

    torch::Tensor decoder_input =
        torch::zeros({features_length.size(0), ContextSize()}, torch::kLong)
            .to(Device())
            .squeeze(1);
    // decoder_input.shape (N, context_size)

    auto decoder_out = RunDecoder(decoder_input).squeeze(1);
    // decoder_out.shape (N, joiner_dim)

    auto logits = RunJoiner(cur_encoder_out, decoder_out);
    // logits.shape (N, vocab_size)

    vocab_size_ = logits.size(-1);
  }

 private:
  int32_t vocab_size_ = -1;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_TRANSDUCER_MODEL_H_
