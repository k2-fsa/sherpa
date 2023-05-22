// sherpa/csrc/online-zipformer2-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_ZIPFORMER2_TRANSDUCER_MODEL_H_
#define SHERPA_CSRC_ONLINE_ZIPFORMER2_TRANSDUCER_MODEL_H_

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa/csrc/online-transducer-model.h"

namespace sherpa {

/** This class implements models from zipformer with causal=True from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py
 * for an instance.
 *
 * You can find the interface and implementation details of the
 * encoder, decoder, and joiner network in the above Python code.
 */
class OnlineZipformer2TransducerModel : public OnlineTransducerModel {
 public:
  explicit OnlineZipformer2TransducerModel(const std::string &filename,
                                           torch::Device device = torch::kCPU);

  torch::IValue StackStates(
      const std::vector<torch::IValue> &states) const override;

  std::vector<torch::IValue> UnStackStates(torch::IValue states) const override;

  torch::IValue GetEncoderInitStates(int32_t batch_size = 1) override;

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

 private:
  torch::jit::Module model_;

  // The following modules are just aliases to modules in model_
  torch::jit::Module encoder_;
  torch::jit::Module decoder_;
  torch::jit::Module joiner_;

  torch::Device device_{"cpu"};

  int32_t context_size_;
  int32_t chunk_size_;
  int32_t chunk_shift_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_ZIPFORMER2_TRANSDUCER_MODEL_H_
