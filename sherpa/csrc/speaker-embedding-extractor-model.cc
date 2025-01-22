// sherpa/csrc/speaker-embedding-extractor-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/speaker-embedding-extractor-model.h"

#include <string>
#include <utility>
#include <vector>

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/speaker-embedding-extractor-model-meta-data.h"

namespace sherpa {

class SpeakerEmbeddingExtractorModel::Impl {
 public:
  explicit Impl(const SpeakerEmbeddingExtractorConfig &config)
      : config_(config) {
    torch::jit::ExtraFilesMap meta_data{
        {"version", {}},
        {"model_type", {}},
    };

    if (config.use_gpu) {
      device_ = torch::Device{torch::kCUDA};
    }

    model_ = torch::jit::load(config.model, device_, meta_data);

    model_.eval();

    if (meta_data.at("model_type") != "3d-speaker") {
      SHERPA_LOGE("Expect model_type '3d-speaker'. Given: '%s'\n",
                  meta_data.at("model_type").c_str());
      SHERPA_EXIT(-1);
    }
  }

  torch::Tensor Compute(torch::Tensor x) {
    return model_.run_method("forward", x).toTensor();
  }

  SpeakerEmbeddingExtractorModelMetaData &GetModelMetadata() {
    return meta_data_;
  }

  const SpeakerEmbeddingExtractorModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  torch::Device Device() const { return device_; }

 private:
  SpeakerEmbeddingExtractorConfig config_;

  torch::jit::Module model_;
  torch::Device device_{torch::kCPU};

  SpeakerEmbeddingExtractorModelMetaData meta_data_;
};

SpeakerEmbeddingExtractorModel::SpeakerEmbeddingExtractorModel(
    const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

SpeakerEmbeddingExtractorModel::~SpeakerEmbeddingExtractorModel() = default;

SpeakerEmbeddingExtractorModelMetaData &
SpeakerEmbeddingExtractorModel::GetModelMetadata() {
  return impl_->GetModelMetadata();
}

torch::Tensor SpeakerEmbeddingExtractorModel::Compute(torch::Tensor x) const {
  return impl_->Compute(x);
}

const SpeakerEmbeddingExtractorModelMetaData &
SpeakerEmbeddingExtractorModel::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

torch::Device SpeakerEmbeddingExtractorModel::Device() const {
  return impl_->Device();
}

}  // namespace sherpa
