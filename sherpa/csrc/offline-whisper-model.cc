// sherpa/csrc/offline-whisper-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/offline-whisper-model.h"

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/offline-whisper-model-meta-data.h"
namespace sherpa {

class OfflineWhisperModel::Impl {
 public:
  Impl(const OfflineModelConfig &config) {
    torch::jit::ExtraFilesMap meta_data{
        {"model_type", {}},
        {"comment", {}},
        {"version", {}},
        {"maintainer", {}},
        {"n_mels", {}},
        {"n_audio_ctx", {}},
        {"n_audio_state", {}},
        {"n_audio_head", {}},
        {"n_audio_layer", {}},
        {"n_vocab", {}},
        {"n_text_ctx", {}},
        {"n_text_state", {}},
        {"n_text_head", {}},
        {"n_text_layer", {}},
        {"sot_sequence", {}},
        {"all_language_tokens", {}},
        {"all_language_codes", {}},
        {"sot", {}},
        {"sot_index", {}},
        {"eot", {}},
        {"blank_id", {}},
        {"is_multilingual", {}},
        {"no_speech", {}},
        {"non_speech_tokens", {}},
        {"transcribe", {}},
        {"translate", {}},
        {"sot_prev", {}},
        {"sot_lm", {}},
        {"no_timestamps", {}},
    };

    if (config.use_gpu) {
      device_ = torch::Device{torch::kCUDA};
    }

    model_ = torch::jit::load(config.whisper.model, device_, meta_data);
    model_.eval();

    if (meta_data.at("model_type") != "whisper" &&
        meta_data.at("model_type") != "Whisper") {
      SHERPA_LOGE("Expect a whisper model. Given: '%s'",
                  meta_data.at("model_type").c_str());
      SHERPA_EXIT(-1);
    }

    SHERPA_LOGE("here");
  }

 private:
  torch::jit::Module model_;
  OfflineWhisperModelMetaData meta_data_;
  torch::Device device_{torch::kCPU};
};

OfflineWhisperModel::OfflineWhisperModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineWhisperModel::~OfflineWhisperModel() = default;

}  // namespace sherpa
