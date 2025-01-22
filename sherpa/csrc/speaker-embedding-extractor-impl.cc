// sherpa/csrc/speaker-embedding-extractor-impl.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/csrc/speaker-embedding-extractor-impl.h"

#include "sherpa/csrc/speaker-embedding-extractor-general-impl.h"

namespace sherpa {

std::unique_ptr<SpeakerEmbeddingExtractorImpl>
SpeakerEmbeddingExtractorImpl::Create(
    const SpeakerEmbeddingExtractorConfig &config) {
  // supports only 3-d speaker for now
  return std::make_unique<SpeakerEmbeddingExtractorGeneralImpl>(config);
}

}  // namespace sherpa
