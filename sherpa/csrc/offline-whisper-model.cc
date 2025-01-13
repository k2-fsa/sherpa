// sherpa/csrc/offline-whisper-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/offline-whisper-model.h"

#include <string>
#include <utility>
#include <vector>

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/offline-whisper-model-meta-data.h"
#include "sherpa/csrc/text-utils.h"
namespace sherpa {

class OfflineWhisperModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) {
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

    InitMetaData(meta_data);

    if (config.debug) {
      SHERPA_LOGE("%s", meta_data_.ToString().c_str());
    }
  }

 private:
  void InitMetaData(const torch::jit::ExtraFilesMap &meta_data) {
    meta_data_.comment = meta_data.at("comment");
    meta_data_.n_mels = atoi(meta_data.at("n_mels").c_str());
    meta_data_.n_audio_ctx = atoi(meta_data.at("n_audio_ctx").c_str());
    meta_data_.n_audio_state = atoi(meta_data.at("n_audio_state").c_str());
    meta_data_.n_audio_head = atoi(meta_data.at("n_audio_head").c_str());
    meta_data_.n_audio_layer = atoi(meta_data.at("n_audio_layer").c_str());
    meta_data_.n_vocab = atoi(meta_data.at("n_vocab").c_str());
    meta_data_.n_text_ctx = atoi(meta_data.at("n_text_ctx").c_str());
    meta_data_.n_text_state = atoi(meta_data.at("n_text_state").c_str());
    meta_data_.n_text_head = atoi(meta_data.at("n_text_head").c_str());
    meta_data_.n_text_layer = atoi(meta_data.at("n_text_layer").c_str());
    meta_data_.sot = atoi(meta_data.at("sot").c_str());
    meta_data_.sot_index = atoi(meta_data.at("sot_index").c_str());
    meta_data_.eot = atoi(meta_data.at("eot").c_str());
    meta_data_.blank_id = atoi(meta_data.at("blank_id").c_str());
    meta_data_.is_multilingual = atoi(meta_data.at("is_multilingual").c_str());
    meta_data_.no_speech = atoi(meta_data.at("no_speech").c_str());
    meta_data_.non_speech_tokens =
        atoi(meta_data.at("non_speech_tokens").c_str());
    meta_data_.transcribe = atoi(meta_data.at("transcribe").c_str());
    meta_data_.translate = atoi(meta_data.at("translate").c_str());
    meta_data_.sot_prev = atoi(meta_data.at("sot_prev").c_str());
    meta_data_.sot_lm = atoi(meta_data.at("sot_lm").c_str());
    meta_data_.no_timestamps = atoi(meta_data.at("no_timestamps").c_str());

    std::vector<std::string> all_language_codes;
    std::vector<int32_t> all_language_tokens;
    SplitStringToIntegers(meta_data.at("sot_sequence"), ",", true,
                          &meta_data_.sot_sequence);

    SplitStringToVector(meta_data.at("all_language_codes"), ",", true,
                        &all_language_codes);

    SplitStringToIntegers(meta_data.at("all_language_tokens"), ",", true,
                          &all_language_tokens);

    for (int32_t i = 0; i < static_cast<int32_t>(all_language_codes.size());
         ++i) {
      meta_data_.lang2id[std::move(all_language_codes[i])] =
          all_language_tokens[i];
    }
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
