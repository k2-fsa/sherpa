// sherpa/csrc/offline-whisper-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_WHISPER_MODEL_H_
#define SHERPA_CSRC_OFFLINE_WHISPER_MODEL_H_

#include <memory>
#include <tuple>
#include <utility>

#include "sherpa/csrc/offline-model-config.h"
#include "sherpa/csrc/offline-whisper-model-meta-data.h"
#include "torch/script.h"

namespace sherpa {

class OfflineWhisperModel {
 public:
  explicit OfflineWhisperModel(const OfflineModelConfig &config);

  ~OfflineWhisperModel();

  const OfflineWhisperModelMetaData &GetModelMetadata() const;

  torch::Device Device() const;

  /**
   * @params features 3-D tensor of shape (N, C, T).
   * @returns Return two tensors:
   *          - n_layer_cross_k_cache, 4-D tensor (num_layers, N, T, C)
   *          - n_layer_cross_v_cache, 4-D tensor (num_layers, N, T, C)
   */
  std::pair<torch::Tensor, torch::Tensor> RunEncoder(
      const torch::Tensor &features) const;

  /*
   *
   * @params tokens A 2-D tensor of shape (N, num_tokens)
   * @param n_layer_self_k_cache  (num_layers, N, dim1,  dim2)
   * @param n_layer_self_v_cache (num_layers, N, dim1, dim2)
   * @param n_layer_cross_k_cache (num_layers, N, T, dim)
   * @param n_layer_cross_v_cache (num_layers, N, T, dim)
   * @param offset A 1-D int32 tensor of shape (N,)
   *
   * @returns Return a tuple of 3 tensors:
   *          - logits,  (N, num_tokens, dim)
   *          - n_layer_self_k_cache,  (num_layers, batch-size, dim1, dim2)
   *          - n_layer_self_v_cache,  (num_layers, batch-size, dim1, dim2)
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RunDecoder(
      const torch::Tensor &tokens, const torch::Tensor &n_layer_self_k_cache,
      const torch::Tensor &n_layer_self_v_cache,
      const torch::Tensor &n_layer_cross_k_cache,
      const torch::Tensor &n_layer_cross_v_cache,
      const torch::Tensor &offset) const;

  torch::Tensor DetectLanguage(
      const torch::Tensor &n_layer_cross_k_cache,
      const torch::Tensor &n_layer_cross_v_cache) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_WHISPER_MODEL_H_
