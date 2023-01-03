// sherpa/csrc/online-conformer-transducer-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_CONFORMER_TRANSDUCER_MODEL_H_
#define SHERPA_CSRC_ONLINE_CONFORMER_TRANSDUCER_MODEL_H_

#include <string>
#include <tuple>
#include <vector>

#include "sherpa/csrc/online-transducer-model.h"

namespace sherpa {

/** This class implements models from pruned_transducer_stateless{2,3,4,5}
 * from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/model.py
 * for an instance.
 *
 * You can find the interface and implementation details of the
 * encoder, decoder, and joiner network in the above Python code.
 */
class OnlineConformerTransducerModel : public OnlineTransducerModel {
 public:
  /** Constructor.
   *
   * @param filename Path to the torchscript model. See
   *                 https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/export.py
   *                 for how to export a model.
   * @param left_context  A value after subsampling
   * @param right_context A value after subsampling
   * @param decode_chunk_size  A value after subsampling
   * @param device  Move the model to this device on loading.
   */
  OnlineConformerTransducerModel(const std::string &filename,
                                 int32_t left_context, int32_t right_context,
                                 int32_t decode_chunk_size,
                                 torch::Device device = torch::kCPU);

  torch::IValue StackStates(
      const std::vector<torch::IValue> &states) const override;

  std::vector<torch::IValue> UnStackStates(torch::IValue states) const override;

  torch::IValue GetEncoderInitStates(int32_t unused = 1) override;

  std::tuple<torch::Tensor, torch::Tensor, torch::IValue> RunEncoder(
      const torch::Tensor &features, const torch::Tensor &features_length,
      const torch::Tensor &num_processed_frames, torch::IValue states) override;

  torch::Tensor RunDecoder(const torch::Tensor &decoder_input) override;

  torch::Tensor RunJoiner(const torch::Tensor &encoder_out,
                          const torch::Tensor &decoder_out) override;

  torch::Device Device() const override { return device_; }

  int32_t ContextSize() const override { return context_size_; }

  int32_t ChunkSize() const override { return chunk_size_; }

  int32_t ChunkShift() const override { return chunk_shift_; }

  // Non virtual methods that used by Python bindings.

  // See
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/conformer.py#L200
  // for what state contains for details.
  // A vector contains two tensors:
  //   - a 3-d tensor: (num_encoder_layers, left_context, encoder_dim)
  //   - a 3-d tensor: (num_encoder_layers, cnn_module_kernel - 1, encoder_dim)
  using State = std::vector<torch::Tensor>;
  torch::IValue StateToIValue(const State &s) const;
  State StateFromIValue(torch::IValue ivalue) const;

 private:
  torch::jit::Module model_;

  // The following modules are just aliases to modules in model_
  torch::jit::Module encoder_;
  torch::jit::Module decoder_;
  torch::jit::Module joiner_;
  torch::jit::Module encoder_proj_;
  torch::jit::Module decoder_proj_;

  torch::Device device_{"cpu"};
  int32_t left_context_;   // after subsampling
  int32_t right_context_;  // after subsampling

  int32_t context_size_;
  int32_t chunk_size_;
  int32_t chunk_shift_;

 private:
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_CONFORMER_TRANSDUCER_MODEL_H_
