// sherpa/csrc/offline-speaker-segmentation-pyannote-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa/csrc/offline-speaker-segmentation-pyannote-model.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/macros.h"
#include "sherpa/csrc/macros.h"
#include "torch/script.h"

namespace sherpa {

class OfflineSpeakerSegmentationPyannoteModel::Impl {
 public:
  explicit Impl(const OfflineSpeakerSegmentationModelConfig &config)
      : config_(config) {
    torch::jit::ExtraFilesMap meta_data{
        {"model_type", {}},
        {"num_speakers", {}},
        {"powerset_max_classes", {}},
        {"num_classes", {}},
        {"sample_rate", {}},
        {"window_size", {}},
        {"receptive_field_size", {}},
        {"receptive_field_shift", {}},
        {"version", {}},
        {"maintainer", {}},
    };

    if (config.use_gpu) {
      device_ = torch::Device{torch::kCUDA};
    }

    model_ = torch::jit::load(config.pyannote.model, device_, meta_data);
    model_.eval();

    if (meta_data.at("model_type") != "pyannote-segmentation-3.0") {
      SHERPA_LOGE(
          "Expected model_type 'pyannote-segmentation-3.0'. Given: '%s'",
          meta_data.at("model_type").c_str());
      SHERPA_EXIT(-1);
    }
    InitMetaData(meta_data);

    if (config.debug) {
      std::ostringstream os;
      os << "----------meta_data for pyannote-segmentation-3.0------\n";
      os << "sample_rate: " << meta_data_.sample_rate << " s\n";
      os << "window_size: " << meta_data_.window_size << " samples\n";
      os << "window_shift: " << meta_data_.window_shift << " samples\n";
      os << "receptive_field_size: " << meta_data_.receptive_field_size
         << " samples\n";
      os << "receptive_field_shift: " << meta_data_.receptive_field_shift
         << " samples\n";
      os << "num_speakers: " << meta_data_.num_speakers << "\n";
      os << "powerset_max_classes: " << meta_data_.powerset_max_classes << "\n";
      os << "num_classes: " << meta_data_.num_classes << "\n";
      SHERPA_LOGE("%s", os.str().c_str());
    }
  }

  const OfflineSpeakerSegmentationPyannoteModelMetaData &GetModelMetaData()
      const {
    return meta_data_;
  }

  torch::Tensor Forward(torch::Tensor x) {
    InferenceMode no_grad;

    return model_.run_method("forward", x).toTensor();
  }

 private:
  void InitMetaData(const torch::jit::ExtraFilesMap &m) {
    meta_data_.sample_rate = atoi(m.at("sample_rate").c_str());
    meta_data_.window_size = atoi(m.at("window_size").c_str());
    meta_data_.window_shift =
        static_cast<int32_t>(0.1 * meta_data_.window_size);
    meta_data_.receptive_field_size =
        atoi(m.at("receptive_field_size").c_str());
    meta_data_.receptive_field_shift =
        atoi(m.at("receptive_field_shift").c_str());
    meta_data_.num_speakers = atoi(m.at("num_speakers").c_str());
    meta_data_.powerset_max_classes =
        atoi(m.at("powerset_max_classes").c_str());
    meta_data_.num_classes = atoi(m.at("num_classes").c_str());
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
