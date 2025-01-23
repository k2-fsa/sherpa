// sherpa/csrc/offline-speaker-diarization-pyannote-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
#define SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa/csrc/fast-clustering.h"
#include "sherpa/csrc/math.h"
#include "sherpa/csrc/offline-speaker-diarization-impl.h"
#include "sherpa/csrc/offline-speaker-segmentation-pyannote-model.h"
#include "sherpa/csrc/speaker-embedding-extractor.h"

namespace sherpa {

namespace {  // NOLINT

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L41
template <class T>
inline void hash_combine(std::size_t *seed, const T &v) {  // NOLINT
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);  // NOLINT
}

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L47
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    std::size_t result = 0;
    hash_combine(&result, pair.first);
    hash_combine(&result, pair.second);
    return result;
  }
};
}  // namespace

using Int32Pair = std::pair<int32_t, int32_t>;

class OfflineSpeakerDiarizationPyannoteImpl
    : public OfflineSpeakerDiarizationImpl {
 public:
  ~OfflineSpeakerDiarizationPyannoteImpl() override = default;

  explicit OfflineSpeakerDiarizationPyannoteImpl(
      const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(config_.segmentation),
        embedding_extractor_(config_.embedding),
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {}

  int32_t SampleRate() const override {
    const auto &meta_data = segmentation_model_.GetModelMetaData();

    return meta_data.sample_rate;
  }

  void SetConfig(const OfflineSpeakerDiarizationConfig &config) override {
    if (!config.clustering.Validate()) {
      SHERPA_LOGE("Invalid clustering config. Skip it");
      return;
    }
    clustering_ = std::make_unique<FastClustering>(config.clustering);
    config_.clustering = config.clustering;
  }

  OfflineSpeakerDiarizationResult Process(
      torch::Tensor samples,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    std::cout << "samples: " << samples.sizes() << "\n";
    if (samples.dim() != 2) {
      SHERPA_LOGE("Support only 2-d tensors. Given: %d",
                  static_cast<int32_t>(samples.dim()));
      return {};
    }

    if (samples.size(0) != 1) {
      SHERPA_LOGE("Support only batch size == 1. Given: %d",
                  static_cast<int32_t>(samples.size(0)));
      return {};
    }

    std::cout << "samples.sizes: " << samples.sizes() << "\n";
    RunSpeakerSegmentationModel(samples);
    return {};
  }

  std::vector<torch::Tensor> RunSpeakerSegmentationModel(
      torch::Tensor samples) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;

    int32_t batch_size = samples.size(0);
    int32_t num_samples = samples.size(1);
    int32_t need_pad = (num_samples < window_size) ||
                       ((num_samples - window_size) % window_shift);
    std::cout << "num_samples < window_size: " << (num_samples - window_size)
              << "\n";
    std::cout << "((num_samples - window_size) % window_shift): "
              << ((num_samples - window_size) % window_shift) << "\n";
    std::cout << "need pad: " << need_pad << "\n";

    if (need_pad) {
      int32_t padding = 0;
      if (num_samples < window_size) {
        padding = window_size - num_samples;
      } else {
        padding = window_shift - ((num_samples - window_size) % window_shift);
      }
      std::cout << "padding size: " << padding << "\n";
      samples = torch::nn::functional::pad(
          samples, torch::nn::functional::PadFuncOptions({0, padding, 0, 0})
                       .mode(torch::kConstant)
                       .value(0));
    }
    int32_t num_segments = (samples.size(1) - window_size) / window_shift + 1;

    if (need_pad || num_segments > 1) {
      samples = samples.as_strided({batch_size, num_segments, window_size},
                                   {samples.size(1), window_shift, 1});
    } else {
      samples = samples.reshape({1, 1, -1});
    }

    samples = samples.reshape({-1, 1, window_size});
    // e.g. samples.sizes: (264, 1, 160000)

    int32_t max_batch_size = 2;
    torch::Tensor log_probs;
    if (samples.size(0) < max_batch_size) {
      log_probs = segmentation_model_.Forward(samples);
    } else {
      std::vector<torch::Tensor> tmp;
      int32_t n = samples.size(0) / max_batch_size;
      for (int32_t i = 0; i < n; ++i) {
        auto this_batch =
            samples.slice(0, i * max_batch_size, (i + 1) * max_batch_size);
        std::cout << i << "/" << n << " -> " << this_batch.sizes() << "\n";
        auto this_log_prob = segmentation_model_.Forward(this_batch);
        std::cout << "    " << this_log_prob.sizes() << "\n";
        tmp.push_back(std::move(this_log_prob));
      }

      if (samples.size(0) % max_batch_size) {
        auto this_batch = samples.slice(0, n * max_batch_size);
        std::cout << n << " -> " << this_batch.sizes() << "\n";
        auto this_log_prob = segmentation_model_.Forward(this_batch);
        std::cout << "    " << this_log_prob.sizes() << "\n";
        tmp.push_back(std::move(this_log_prob));
      }

      log_probs = torch::cat(tmp, 0);
    }
    // e.g. log_probs.sizes: (264, 589, 7)
    std::cout << "log_probs.sizes: " << log_probs.sizes() << "\n";

    // TODO(fangjun): Limit the batch size here

    return {};
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  OfflineSpeakerSegmentationPyannoteModel segmentation_model_;
  SpeakerEmbeddingExtractor embedding_extractor_;
  std::unique_ptr<FastClustering> clustering_;
};

}  // namespace sherpa
#endif  // SHERPA_NNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
