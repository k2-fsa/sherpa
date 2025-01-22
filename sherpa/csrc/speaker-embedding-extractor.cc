// sherpa/csrc/speaker-embedding-extractor.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa/csrc/speaker-embedding-extractor.h"

#include <vector>

#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/speaker-embedding-extractor-impl.h"

namespace sherpa {

void SpeakerEmbeddingExtractorConfig::Register(ParseOptions *po) {
  po->Register("model", &model, "Path to the speaker embedding model.");
  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("use_gpu", &use_gpu, "true to gpu.");
}

bool SpeakerEmbeddingExtractorConfig::Validate() const {
  if (model.empty()) {
    SHERPA_LOGE("Please provide a speaker embedding extractor model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_LOGE("speaker embedding extractor model: '%s' does not exist",
                model.c_str());
    return false;
  }

  return true;
}

std::string SpeakerEmbeddingExtractorConfig::ToString() const {
  std::ostringstream os;

  os << "SpeakerEmbeddingExtractorConfig(";
  os << "model=\"" << model << "\", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "use_gpu=" << (use_gpu ? "True" : "False") << ")";

  return os.str();
}

SpeakerEmbeddingExtractor::SpeakerEmbeddingExtractor(
    const SpeakerEmbeddingExtractorConfig &config)
    : impl_(SpeakerEmbeddingExtractorImpl::Create(config)) {}

SpeakerEmbeddingExtractor::~SpeakerEmbeddingExtractor() = default;

int32_t SpeakerEmbeddingExtractor::Dim() const { return impl_->Dim(); }

std::unique_ptr<OfflineStream> SpeakerEmbeddingExtractor::CreateStream() const {
  return impl_->CreateStream();
}

torch::Tensor SpeakerEmbeddingExtractor::Compute(OfflineStream *s) const {
  return impl_->Compute(s);
}

torch::Tensor SpeakerEmbeddingExtractor::Compute(OfflineStream **ss,
                                                 int32_t n) const {
  return impl_->Compute(ss, n);
}

}  // namespace sherpa
