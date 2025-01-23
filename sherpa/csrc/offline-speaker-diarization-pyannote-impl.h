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
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {
    InitPowersetMapping();
    std::cout << "powerset_mapping: " << powerset_mapping_ << "\n";
  }

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
    torch::Tensor log_probs = RunSpeakerSegmentationModel(samples);
    std::cout << "log_probs.sizes: " << log_probs.sizes() << "\n";
    // A chunk is a window_size samples
    // log_probs: (num_chunks, num_frames, 7)
    // where 7 is the num_powerset_classes

    torch::Tensor labels = ToMultiLabel(log_probs);
    std::cout << "labels.sizes: " << labels.sizes() << "\n";

    return {};
  }

  torch::Tensor RunSpeakerSegmentationModel(torch::Tensor samples) const {
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

    return log_probs;
  }

  // see
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/utils/powerset.py#L103
  torch::Tensor ToMultiLabel(torch::Tensor log_probs) const {
    int32_t num_classes = powerset_mapping_.size(0);
    auto powerset_probs = torch::nn::functional::one_hot(
                              torch::argmax(log_probs, -1), num_classes)
                              .to(torch::kFloat);
    std::cout << "powerset_probs.sizes: " << powerset_probs.sizes() << "\n";
    auto labels = torch::matmul(powerset_probs, powerset_mapping_);
    std::cout << "labels.sizes: " << labels.sizes() << "\n";
    return labels;
  }

 private:
  void InitPowersetMapping() {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t num_classes = meta_data.num_classes;
    int32_t powerset_max_classes = meta_data.powerset_max_classes;
    int32_t num_speakers = meta_data.num_speakers;

    powerset_mapping_ =
        torch::zeros({num_classes, num_speakers}, torch::kFloat);
    auto acc = powerset_mapping_.accessor<float, 2>();

    int32_t k = 1;
    for (int32_t i = 1; i <= powerset_max_classes; ++i) {
      if (i == 1) {
        for (int32_t j = 0; j != num_speakers; ++j, ++k) {
          acc[k][j] = 1;
        }
      } else if (i == 2) {
        for (int32_t j = 0; j != num_speakers; ++j) {
          for (int32_t m = j + 1; m < num_speakers; ++m, ++k) {
            acc[k][j] = 1;
            acc[k][m] = 1;
          }
        }
      } else {
        SHERPA_LOGE("powerset_max_classes = %d is currently not supported!", i);
        SHERPA_EXIT(-1);
      }
    }
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  OfflineSpeakerSegmentationPyannoteModel segmentation_model_;
  SpeakerEmbeddingExtractor embedding_extractor_;
  std::unique_ptr<FastClustering> clustering_;
  torch::Tensor powerset_mapping_;  // 2-d float tensor
  /*
 0  0  0   // 0
 1  0  0   // 1
 0  1  0   // 2
 0  0  1   // 3
 1  1  0   // 4
 1  0  1   // 5
 0  1  1   // 6
 */
};

}  // namespace sherpa
#endif  // SHERPA_NNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
