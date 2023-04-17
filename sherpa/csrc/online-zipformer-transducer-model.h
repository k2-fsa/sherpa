// sherpa/csrc/online-lstm-transducer-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_H_
#define SHERPA_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_H_
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa/csrc/online-transducer-model.h"

namespace sherpa {
/** This class implements models from pruned_transducer_stateless7_streaming
 * from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py
 * for an instance.
 *
 * You can find the interface and implementation details of the
 * encoder, decoder, and joiner network in the above Python code.
 */
class OnlineZipformerTransducerModel : public OnlineTransducerModel {
 public:
  /** Constructor.
   *
   * @param filename Path to the torchscript model. See
   *                 https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/jit_trace_export.py
   *                 for how to export a model.
   * @param decode_chunk_size  Number of frames before subsampling
   * @param device  Move the model to this device on loading.
   */
  OnlineZipformerTransducerModel(const std::string &encoder_filename,
                                 const std::string &decoder_filename,
                                 const std::string &joiner_filename,
                                 torch::Device device = torch::kCPU);

  explicit OnlineZipformerTransducerModel(const std::string &filename,
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
  // true if the model is from torch.jit.trace()
  bool from_torch_jit_trace_;

  const char *encoder_forward_method_name_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_H_
