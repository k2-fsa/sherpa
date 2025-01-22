// sherpa/csrc/speaker-embedding-extractor-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_
#define SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa/csrc/speaker-embedding-extractor.h"

namespace sherpa {

class SpeakerEmbeddingExtractorImpl {
 public:
  virtual ~SpeakerEmbeddingExtractorImpl() = default;

  static std::unique_ptr<SpeakerEmbeddingExtractorImpl> Create(
      const SpeakerEmbeddingExtractorConfig &config);

  virtual int32_t Dim() const = 0;

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual torch::Tensor Compute(OfflineStream *s) const = 0;

  virtual torch::Tensor Compute(OfflineStream **s, int32_t n) const = 0;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_
