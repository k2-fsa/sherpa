// sherpa/csrc/speaker-embedding-extractor-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_META_DATA_H_
#define SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_META_DATA_H_

#include <cstdint>
#include <string>

namespace sherpa {

struct SpeakerEmbeddingExtractorModelMetaData {
  int32_t output_dim = 0;
  int32_t sample_rate = 0;

  // for wespeaker models, it is 0;
  // for 3d-speaker models, it is 1
  int32_t normalize_samples = 1;

  // Chinese, English, etc.
  std::string language;

  // for 3d-speaker, it is global-mean
  std::string feature_normalize_type;
};

}  // namespace sherpa
#endif  // SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_META_DATA_H_
