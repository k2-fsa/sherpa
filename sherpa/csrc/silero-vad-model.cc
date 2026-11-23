// sherpa/csrc/silero-vad-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/csrc/silero-vad-model.h"

#include "sherpa/cpp_api/macros.h"
#include "sherpa/csrc/macros.h"
namespace sherpa {

class SileroVadModel::Impl {
 public:
  explicit Impl(const VadModelConfig &config) : config_(config) {
    torch::jit::ExtraFilesMap meta_data{
        {"version", {}},
        {"model_type", {}},
    };

    if (config.use_gpu) {
      device_ = torch::Device{torch::kCUDA};
    }

    model_ = torch::jit::load(config.silero_vad.model, device_, meta_data);

    model_.eval();

    if (meta_data.at("model_type") != "silero_vad") {
      SHERPA_LOGE("Expect model_type 'silero_vad'. Given: '%s'\n",
                  meta_data.at("model_type").c_str());
      SHERPA_EXIT(-1);
    }

    if (meta_data.at("version") != "4") {
      SHERPA_LOGE("It supports only silero_vad v4. Given version: '%s'\n",
                  meta_data.at("version").c_str());
      SHERPA_EXIT(-1);
    }
  }

  torch::Device Device() const { return device_; }

  torch::Tensor Run(torch::Tensor samples) {
    InferenceMode no_grad;

    torch::Tensor sample_rate = torch::tensor(
        {config_.sample_rate}, torch::dtype(torch::kInt).device(device_));

    int32_t window_size = 512;
    return model_
        .run_method("audio_forward", samples, config_.sample_rate, window_size)
        .toTensor();
  }

 private:
  torch::jit::Module model_;
  torch::Device device_{torch::kCPU};
  VadModelConfig config_;
};

SileroVadModel::SileroVadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

SileroVadModel::~SileroVadModel() = default;

torch::Device SileroVadModel::Device() const { return impl_->Device(); }

torch::Tensor SileroVadModel::Run(torch::Tensor samples) const {
  return impl_->Run(samples);
}

}  // namespace sherpa
