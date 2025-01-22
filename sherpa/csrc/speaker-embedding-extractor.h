// sherpa/csrc/speaker-embedding-extractor.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_
#define SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa/cpp_api/offline-stream.h"
#include "sherpa/cpp_api/parse-options.h"

namespace sherpa {

struct SpeakerEmbeddingExtractorConfig {
  std::string model;
  bool use_gpu = false;
  bool debug = false;

  SpeakerEmbeddingExtractorConfig() = default;
  SpeakerEmbeddingExtractorConfig(const std::string &model, bool use_gpu,
                                  bool debug)
      : model(model), use_gpu(use_gpu), debug(debug) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class SpeakerEmbeddingExtractorImpl;

class SpeakerEmbeddingExtractor {
 public:
  explicit SpeakerEmbeddingExtractor(
      const SpeakerEmbeddingExtractorConfig &config);

  template <typename Manager>
  SpeakerEmbeddingExtractor(Manager *mgr,
                            const SpeakerEmbeddingExtractorConfig &config);

  ~SpeakerEmbeddingExtractor();

  // Return the dimension of the embedding
  int32_t Dim() const;

  // Create a stream to accept audio samples and compute features
  std::unique_ptr<OfflineStream> CreateStream() const;

  // Compute the speaker embedding from the available unprocessed features
  // of the given stream
  //
  // You have to ensure IsReady(s) returns true before you call this method.
  torch::Tensor Compute(OfflineStream *s) const;

  torch::Tensor Compute(OfflineStream **ss, int32_t n) const;

 private:
  std::unique_ptr<SpeakerEmbeddingExtractorImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_
