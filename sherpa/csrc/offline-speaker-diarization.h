// sherpa/csrc/offline-speaker-diarization.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_
#define SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_

#include <functional>
#include <memory>
#include <string>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/fast-clustering-config.h"
#include "sherpa/csrc/offline-speaker-diarization-result.h"
#include "sherpa/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa/csrc/speaker-embedding-extractor.h"

namespace sherpa {

struct OfflineSpeakerDiarizationConfig {
  OfflineSpeakerSegmentationModelConfig segmentation;
  SpeakerEmbeddingExtractorConfig embedding;
  FastClusteringConfig clustering;

  // if a segment is less than this value, then it is discarded
  float min_duration_on = 0.3;  // in seconds

  // if the gap between to segments of the same speaker is less than this value,
  // then these two segments are merged into a single segment.
  // We do this recursively.
  float min_duration_off = 0.5;  // in seconds

  OfflineSpeakerDiarizationConfig() = default;

  OfflineSpeakerDiarizationConfig(
      const OfflineSpeakerSegmentationModelConfig &segmentation,
      const SpeakerEmbeddingExtractorConfig &embedding,
      const FastClusteringConfig &clustering, float min_duration_on,
      float min_duration_off)
      : segmentation(segmentation),
        embedding(embedding),
        clustering(clustering),
        min_duration_on(min_duration_on),
        min_duration_off(min_duration_off) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class OfflineSpeakerDiarizationImpl;

using OfflineSpeakerDiarizationProgressCallback = std::function<int32_t(
    int32_t processed_chunks, int32_t num_chunks, void *arg)>;

class OfflineSpeakerDiarization {
 public:
  explicit OfflineSpeakerDiarization(
      const OfflineSpeakerDiarizationConfig &config);

  ~OfflineSpeakerDiarization();

  // Expected sample rate of the input audio samples
  int32_t SampleRate() const;

  // Note: Only config.clustering is used. All other fields in config are
  // ignored
  void SetConfig(const OfflineSpeakerDiarizationConfig &config);

  // @param samples 2-D tensor of shape (batch_size, num_samples)
  OfflineSpeakerDiarizationResult Process(
      torch::Tensor samples,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const;

 private:
  std::unique_ptr<OfflineSpeakerDiarizationImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_H_
