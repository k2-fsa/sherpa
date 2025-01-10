// sherpa/csrc/offline-sense-voice-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/csrc/offline-sense-voice-model.h"

#include <utility>

#include "sherpa/cpp_api/macros.h"
#include "sherpa/csrc/macros.h"

namespace sherpa {

static std::vector<float> ToFloat(const std::string &s) {
  const float *p = reinterpret_cast<const float *>(s.data());
  int32_t n = s.size() / 4;

  // assume little endian
  return {p, p + n};
}

class OfflineSenseVoiceModel::Impl {
 public:
  Impl(const OfflineModelConfig &config) {
    torch::jit::ExtraFilesMap meta_data{
        {"model_type", ""},        {"lfr_window_size", ""},
        {"lfr_window_shift", ""},  {"neg_mean", ""},
        {"inv_stddev", ""},        {"vocab_size", ""},
        {"normalize_samples", ""}, {"version", ""},
        {"model_author", ""},      {"maintainer", ""},
        {"lang_auto", ""},         {"lang_zh", ""},
        {"lang_en", ""},           {"lang_yue", ""},
        {"lang_ja", ""},           {"lang_ko", ""},
        {"lang_nospeech", ""},     {"with_itn", ""},
        {"without_itn", ""},       {"url", ""},
    };
    if (config.use_gpu) {
      device_ = torch::Device{torch::kCUDA};
    }

    model_ = torch::jit::load(config.sense_voice.model, device_, meta_data);
    model_.eval();

    if (meta_data.at("model_type") != "SenseVoiceSmall") {
      SHERPA_LOGE("Expect a SenseVoiceSmall model. Given: '%s'",
                  meta_data.at("model_type").c_str());
      SHERPA_EXIT(-1);
    }

    InitMetaData(meta_data);

    if (config.debug) {
      SHERPA_LOGE("%s", meta_data_.ToString().c_str());
    }
  }

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  torch::Device Device() const { return device_; }

  std::pair<torch::Tensor, torch::Tensor> RunForward(
      const torch::Tensor &features, const torch::Tensor &features_length,
      const torch::Tensor &language, const torch::Tensor &use_itn) {
    InferenceMode no_grad;

    auto outputs =
        model_
            .run_method("forward", features, features_length, language, use_itn)
            .toTuple();

    auto logits = outputs->elements()[0].toTensor();
    auto logits_length = outputs->elements()[1].toTensor();

    return {logits, logits_length};
  }

 private:
  void InitMetaData(const torch::jit::ExtraFilesMap &meta_data) {
    meta_data_.with_itn_id = atoi(meta_data.at("with_itn").c_str());
    meta_data_.without_itn_id = atoi(meta_data.at("without_itn").c_str());
    meta_data_.window_size = atoi(meta_data.at("lfr_window_size").c_str());
    meta_data_.window_shift = atoi(meta_data.at("lfr_window_shift").c_str());
    meta_data_.vocab_size = atoi(meta_data.at("vocab_size").c_str());
    meta_data_.normalize_samples =
        atoi(meta_data.at("normalize_samples").c_str());

    meta_data_.lang2id["auto"] = atoi(meta_data.at("lang_auto").c_str());
    meta_data_.lang2id["zh"] = atoi(meta_data.at("lang_zh").c_str());
    meta_data_.lang2id["en"] = atoi(meta_data.at("lang_en").c_str());
    meta_data_.lang2id["yue"] = atoi(meta_data.at("lang_yue").c_str());
    meta_data_.lang2id["ko"] = atoi(meta_data.at("lang_ko").c_str());
    meta_data_.lang2id["ja"] = atoi(meta_data.at("lang_ja").c_str());

    auto neg_mean = ToFloat(meta_data.at("neg_mean"));
    auto inv_stddev = ToFloat(meta_data.at("inv_stddev"));

    meta_data_.neg_mean =
        torch::from_blob(neg_mean.data(),
                         {1, static_cast<int32_t>(neg_mean.size())},
                         torch::kFloat32)
            .clone();

    meta_data_.inv_stddev =
        torch::from_blob(inv_stddev.data(),
                         {1, static_cast<int32_t>(inv_stddev.size())},
                         torch::kFloat32)
            .clone();
  }

 private:
  torch::jit::Module model_;
  OfflineSenseVoiceModelMetaData meta_data_;
  torch::Device device_{torch::kCPU};
};

OfflineSenseVoiceModel::OfflineSenseVoiceModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineSenseVoiceModel::~OfflineSenseVoiceModel() = default;

const OfflineSenseVoiceModelMetaData &OfflineSenseVoiceModel::GetModelMetadata()
    const {
  return impl_->GetModelMetadata();
}

torch::Device OfflineSenseVoiceModel::Device() const { return impl_->Device(); }

std::pair<torch::Tensor, torch::Tensor> OfflineSenseVoiceModel::RunForward(
    const torch::Tensor &features, const torch::Tensor &features_length,
    const torch::Tensor &language, const torch::Tensor &use_itn) {
  return impl_->RunForward(features, features_length, language, use_itn);
}

}  // namespace sherpa
