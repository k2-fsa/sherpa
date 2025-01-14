// sherpa/cpp_api/offline-recognizer-whisper-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_

#include <memory>

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/offline-whisper-model.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

static OfflineRecognitionResult Convert(const std::vector<int32_t> &tokens,
                                        const SymbolTable &sym_table) {
  OfflineRecognitionResult r;
  r.tokens.reserve(tokens.size());

  std::string text;
  for (auto i : tokens) {
    auto sym = sym_table[i];
    text.append(sym);

    r.tokens.push_back(std::move(sym));
  }
  r.text = std::move(text);

  return r;
}

class OfflineRecognizerWhisperImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWhisperImpl(const OfflineRecognizerConfig &config)
      : config_(config), symbol_table_(config.model.tokens) {
    symbol_table_.ApplyBase64Decode();

    model_ = std::make_unique<OfflineWhisperModel>(config.model);

    config_.feat_config.normalize_samples = true;

    auto whisper_opts = kaldifeat::WhisperFbankOptions();
    whisper_opts.num_mels = model_->GetModelMetadata().n_mels;

    whisper_ = std::make_unique<kaldifeat::WhisperFbank>(whisper_opts);
  }

  std::unique_ptr<OfflineStream> CreateStream() override {
    return std::make_unique<OfflineStream>(whisper_.get(), config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) override {
    InferenceMode no_grad;
    if (n == 1) {
      DecodeStream(ss[0]);
      return;
    }
  }

 private:
  void DecodeStream(OfflineStream *s) {
    auto device = model_->Device();

    torch::Tensor features = s->GetFeatures();
    features = PadOrTrimFeatures(features);
    features = features.t().unsqueeze(0).to(device);

    torch::Tensor n_layer_cross_k_cache;
    torch::Tensor n_layer_cross_v_cache;

    std::tie(n_layer_cross_k_cache, n_layer_cross_v_cache) =
        model_->RunEncoder(features);

    auto meta_data = model_->GetModelMetadata();
    auto sot_sequence = meta_data.sot_sequence;
    sot_sequence.push_back(meta_data.no_timestamps);

    if (meta_data.is_multilingual) {
      // sot_sequence: [sot, language, task, notimestamp]
      auto language = config_.model.whisper.language;
      if (!language.empty()) {
        if (!meta_data.lang2id.count(language)) {
          SHERPA_LOG(FATAL) << "language '" << language << " is not valid";
        }

        sot_sequence[1] = meta_data.lang2id.at(language);
      } else {
        if (config_.model.debug) {
          SHERPA_LOGE("Begin to detect language");
        }
        sot_sequence[1] = model_->DetectLanguage(n_layer_cross_k_cache,
                                                 n_layer_cross_v_cache);
        if (config_.model.debug) {
          SHERPA_LOGE("Detected language: %s",
                      meta_data.id2lang.at(sot_sequence[1]).c_str());
        }
      }

      if (config_.model.whisper.task == "translate") {
        sot_sequence[2] = meta_data.translate;
      }
    }

    torch::Tensor tokens =
        torch::from_blob(sot_sequence.data(),
                         {1, static_cast<int32_t>(sot_sequence.size())},
                         torch::kLong)
            .to(device);

    torch::Tensor logits;

    torch::Tensor n_layer_self_k_cache =
        torch::zeros({meta_data.n_text_layer, 1, meta_data.n_text_ctx,
                      meta_data.n_text_state},
                     torch::dtype(torch::kFloat).device(device));

    torch::Tensor n_layer_self_v_cache =
        torch::zeros({meta_data.n_text_layer, 1, meta_data.n_text_ctx,
                      meta_data.n_text_state},
                     torch::dtype(torch::kFloat).device(device));

    torch::Tensor offset =
        torch::zeros({1}, torch::dtype(torch::kInt).device(device));

    std::tie(logits, n_layer_self_k_cache, n_layer_self_v_cache) =
        model_->RunDecoder(tokens, n_layer_self_k_cache, n_layer_self_v_cache,
                           n_layer_cross_k_cache, n_layer_cross_v_cache,
                           offset);

    torch::Tensor eot = torch::tensor(
        {meta_data.eot}, torch::dtype(torch::kLong).device(device));

    torch::Tensor results =
        torch::full({1, meta_data.n_text_ctx}, meta_data.eot,
                    torch::dtype(torch::kLong).device(device));

    int32_t i;
    for (i = 0; i < meta_data.n_text_ctx; ++i) {
      tokens = logits.slice(1, -1).argmax(-1);
      if ((tokens == eot).sum().item().toInt() == 1) {
        break;
      }
      results.slice(1, i, i + 1) = tokens;
      offset.add_(logits.size(1));

      std::tie(logits, n_layer_self_k_cache, n_layer_self_v_cache) =
          model_->RunDecoder(tokens, n_layer_self_k_cache, n_layer_self_v_cache,
                             n_layer_cross_k_cache, n_layer_cross_v_cache,
                             offset);
    }
    results = results.slice(1, 0, i).cpu();

    std::vector<int32_t> token_ids = {
        results.data_ptr<int64_t>(),
        results.data_ptr<int64_t>() + results.numel()};

    s->SetResult(Convert(token_ids, symbol_table_));
  }

 private:
  void WarmUp() {
    SHERPA_LOG(INFO) << "WarmUp begins";

    SHERPA_LOG(INFO) << "WarmUp ended";
  }

  torch::Tensor PadOrTrimFeatures(const torch::Tensor &feat) {
    auto features = feat;
    int32_t target_len = 3000;
    int32_t src_len = features.size(0);
    if (src_len > target_len) {
      SHERPA_LOGE(
          "\nInput audio is too long (about %.3f seconds). Only the first %d "
          "seconds are used.",
          src_len * 0.01, static_cast<int32_t>(target_len * 0.01));
      features = features.slice(0, 0, target_len);
    } else if (src_len < target_len) {
      int32_t padding = target_len - src_len;
      features = torch::nn::functional::pad(
          features, torch::nn::functional::PadFuncOptions({0, 0, 0, padding})
                        .mode(torch::kConstant)
                        .value(0));
    }

    return features;
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<kaldifeat::WhisperFbank> whisper_;
  std::unique_ptr<OfflineWhisperModel> model_;
};
}  // namespace sherpa
#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
