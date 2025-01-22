// sherpa/csrc/offline-speaker-segmentation-pyannote-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa/csrc/offline-speaker-segmentation-pyannote-model.h"

#include <string>
#include <utility>
#include <vector>

#include "torch/script.h"

namespace sherpa {

class OfflineSpeakerSegmentationPyannoteModel::Impl {
 public:
  explicit Impl(const OfflineSpeakerSegmentationModelConfig &config)
      : config_(config) {
    torch::jit::ExtraFilesMap meta_data{
        {"model_type", {}},
    };

    if (config.use_gpu) {
      device_ = torch::Device{torch::kCUDA};
    }

    model_ = torch::jit::load(config.pyannote.model, device_, meta_data);
    model_.eval();
  }

  const OfflineSpeakerSegmentationPyannoteModelMetaData &GetModelMetaData()
      const {
    return meta_data_;
  }

  torch::Tensor Forward(torch::Tensor x) {
    return model_.run_method("forward", x).toTensor();
  }

 private:
  OfflineSpeakerSegmentationModelConfig config_;
  OfflineSpeakerSegmentationPyannoteModelMetaData meta_data_;
  torch::jit::Module model_;
  torch::Device device_{torch::kCPU};
};

OfflineSpeakerSegmentationPyannoteModel::
    OfflineSpeakerSegmentationPyannoteModel(
        const OfflineSpeakerSegmentationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineSpeakerSegmentationPyannoteModel::
    ~OfflineSpeakerSegmentationPyannoteModel() = default;

const OfflineSpeakerSegmentationPyannoteModelMetaData &
OfflineSpeakerSegmentationPyannoteModel::GetModelMetaData() const {
  return impl_->GetModelMetaData();
}

torch::Tensor OfflineSpeakerSegmentationPyannoteModel::Forward(
    torch::Tensor x) const {
  return impl_->Forward(x);
}

}  // namespace sherpa
