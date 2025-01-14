// sherpa/csrc/offline-whisper-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/offline-whisper-model.h"

#include <string>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/macros.h"
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

  const OfflineWhisperModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  torch::Device Device() const { return device_; }

  std::pair<torch::Tensor, torch::Tensor> RunEncoder(
      const torch::Tensor &features) {
    InferenceMode no_grad;

    auto outputs = model_.run_method("run_encoder", features).toTuple();

    auto n_layer_cross_k_cache = outputs->elements()[0].toTensor();
    auto n_layer_cross_v_cache = outputs->elements()[1].toTensor();

    return {n_layer_cross_k_cache, n_layer_cross_v_cache};
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RunDecoder(
      const torch::Tensor &tokens, torch::Tensor n_layer_self_k_cache,
      torch::Tensor n_layer_self_v_cache, torch::Tensor n_layer_cross_k_cache,
      torch::Tensor n_layer_cross_v_cache, const torch::Tensor &offset) {
    InferenceMode no_grad;

    auto outputs = model_
                       .run_method("run_decoder", tokens, n_layer_self_k_cache,
                                   n_layer_self_v_cache, n_layer_cross_k_cache,
                                   n_layer_cross_v_cache, offset)
                       .toTuple();

    auto logits = outputs->elements().vec()[0].toTensor();
    n_layer_self_k_cache = outputs->elements().vec()[1].toTensor();
    n_layer_self_v_cache = outputs->elements().vec()[2].toTensor();

    return std::make_tuple(logits, n_layer_self_k_cache, n_layer_self_v_cache);
  }

  int32_t DetectLanguage(const torch::Tensor &n_layer_cross_k_cache,
                         const torch::Tensor &n_layer_cross_v_cache) {
    InferenceMode no_grad;

    torch::Tensor tokens =
        torch::tensor({meta_data_.sot},
                      torch::dtype(torch::kInt).device(device_))
            .unsqueeze(0);

    torch::Tensor offset =
        torch::zeros({1}, torch::dtype(torch::kInt).device(device_));

    torch::Tensor n_layer_self_k_cache =
        torch::zeros({meta_data_.n_text_layer, 1, meta_data_.n_text_ctx,
                      meta_data_.n_text_state},
                     torch::dtype(torch::kFloat).device(device_));

    torch::Tensor n_layer_self_v_cache =
        torch::zeros({meta_data_.n_text_layer, 1, meta_data_.n_text_ctx,
                      meta_data_.n_text_state},
                     torch::dtype(torch::kFloat).device(device_));

    auto out = RunDecoder(tokens, n_layer_self_k_cache, n_layer_self_v_cache,
                          n_layer_cross_k_cache, n_layer_cross_v_cache, offset);
    auto logits = std::get<0>(out).squeeze();

    torch::Tensor all_languages_id =
        torch::tensor(meta_data_.all_languages_id,
                      torch::dtype(torch::kLong).device(device_));
    torch::Tensor mask =
        torch::ones(logits.size(0), torch::dtype(torch::kLong).device(device_));
    mask.index_put_({all_languages_id}, 0);

    torch::Tensor non_language_indexes = mask.nonzero().squeeze();

    logits.index_put_({non_language_indexes}, -1e30);

    return logits.argmax(-1).item().toInt();
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
    SplitStringToIntegers(meta_data.at("sot_sequence"), ",", true,
                          &meta_data_.sot_sequence);

    SplitStringToVector(meta_data.at("all_language_codes"), ",", true,
                        &all_language_codes);

    SplitStringToIntegers(meta_data.at("all_language_tokens"), ",", true,
                          &meta_data_.all_languages_id);

    for (int32_t i = 0; i < static_cast<int32_t>(all_language_codes.size());
         ++i) {
      meta_data_.lang2id[all_language_codes[i]] =
          meta_data_.all_languages_id[i];

      meta_data_.id2lang[meta_data_.all_languages_id[i]] =
          std::move(all_language_codes[i]);
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

const OfflineWhisperModelMetaData &OfflineWhisperModel::GetModelMetadata()
    const {
  return impl_->GetModelMetadata();
}

torch::Device OfflineWhisperModel::Device() const { return impl_->Device(); }

std::pair<torch::Tensor, torch::Tensor> OfflineWhisperModel::RunEncoder(
    const torch::Tensor &features) const {
  return impl_->RunEncoder(features);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
OfflineWhisperModel::RunDecoder(const torch::Tensor &tokens,
                                const torch::Tensor &n_layer_self_k_cache,
                                const torch::Tensor &n_layer_self_v_cache,
                                const torch::Tensor &n_layer_cross_k_cache,
                                const torch::Tensor &n_layer_cross_v_cache,
                                const torch::Tensor &offset) const {
  return impl_->RunDecoder(tokens, n_layer_self_k_cache, n_layer_self_v_cache,
                           n_layer_cross_k_cache, n_layer_cross_v_cache,
                           offset);
}

int32_t OfflineWhisperModel::DetectLanguage(
    const torch::Tensor &n_layer_cross_k_cache,
    const torch::Tensor &n_layer_cross_v_cache) const {
  return impl_->DetectLanguage(n_layer_cross_k_cache, n_layer_cross_v_cache);
}

}  // namespace sherpa
