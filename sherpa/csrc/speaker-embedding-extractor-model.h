// sherpa/csrc/speaker-embedding-extractor-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_H_
#define SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_H_

#include <memory>

#include "sherpa/csrc/speaker-embedding-extractor-model-meta-data.h"
#include "sherpa/csrc/speaker-embedding-extractor.h"
#include "torch/script.h"

namespace sherpa {

class SpeakerEmbeddingExtractorModel {
 public:
  explicit SpeakerEmbeddingExtractorModel(
      const SpeakerEmbeddingExtractorConfig &config);

  ~SpeakerEmbeddingExtractorModel();

  SpeakerEmbeddingExtractorModelMetaData &GetModelMetadata();
  const SpeakerEmbeddingExtractorModelMetaData &GetModelMetadata() const;

  /**
   * @param x A float32 tensor of shape (N, T, C)
   * @return A float32 tensor of shape (N, C)
   */
  torch::Tensor Compute(torch::Tensor x) const;

  torch::Device Device() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_H_
