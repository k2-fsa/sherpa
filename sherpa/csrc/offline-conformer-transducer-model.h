// sherpa/csrc/offline-conformer-transducer-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_CONFORMER_TRANSDUCER_MODEL_H_
#define SHERPA_CSRC_OFFLINE_CONFORMER_TRANSDUCER_MODEL_H_

#include <string>
#include <utility>

#include "sherpa/csrc/offline-transducer-model.h"

namespace sherpa {

/** This class implements models from pruned_transducer_statelessX
 * where X>=2 from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/model.py
 * for an instance.
 *
 * You can find the interface and implementation details of the
 * encoder, decoder, and joiner network in the above Python code.
 */
class OfflineConformerTransducerModel : public OfflineTransducerModel {
 public:
  explicit OfflineConformerTransducerModel(const std::string &filename,
                                           torch::Device device = torch::kCPU);

  /**
   * See
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/conformer.py#L127
   * for the interface of the encoder module.
   * Note that we use the default value warmup 1.0 here.
   *
   * Also, the output is transformed by using the projection module from
   * the joiner, please see
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/joiner.py#L34
   */
  std::pair<torch::Tensor, torch::Tensor> RunEncoder(
      const torch::Tensor &features,
      const torch::Tensor &features_length) override;

  // It returns the projected decoder out.
  /**
   * See
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/decoder.py#L82
   * for the interface of the decoder module.
   *
   * We set `need_pad` to false inside this method.
   *
   * Also, the output is transformed by using the projection module from
   * the joiner, please see
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/joiner.py#L35
   */
  torch::Tensor RunDecoder(const torch::Tensor &decoder_input) override;

  /**
   * See
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/joiner.py#L38
   * for the interface of the joiner module.
   *
   * We set `project_input` to false inside this method.
   *
   * Both inputs are of shape (N, joiner_dim). The output shape is
   * (N, vocab_size).
   */
  torch::Tensor RunJoiner(const torch::Tensor &encoder_out,
                          const torch::Tensor &decoder_out) override;

  torch::Device Device() const override { return device_; }

  /* See
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/decoder.py#L67
   * for the definition and usage of context_size.
   */
  int32_t ContextSize() const override { return context_size_; }

 private:
  torch::jit::Module model_;

  // The following modules are just aliases to modules in model_
  torch::jit::Module encoder_;
  torch::jit::Module decoder_;
  torch::jit::Module joiner_;
  torch::jit::Module encoder_proj_;
  torch::jit::Module decoder_proj_;

  torch::Device device_{"cpu"};
  int32_t context_size_;
};

}  // namespace sherpa

#endif  //  SHERPA_CSRC_OFFLINE_CONFORMER_TRANSDUCER_MODEL_H_
