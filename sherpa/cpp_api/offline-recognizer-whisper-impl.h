// sherpa/cpp_api/offline-recognizer-whisper-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

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

    auto device = model_->Device();

#if 0
    // TODO(fangjun): Figure out why this branch does not work.
    // All wave files are decoded into the same result like the first wave file
    std::vector<torch::Tensor> features_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      auto features = ss[i]->GetFeatures();
      features = PadOrTrimFeatures(features);
      features_vec[i] = PadOrTrimFeatures(features);
    }

    auto features = torch::stack(features_vec, 0).to(device).permute({0, 2, 1});

    torch::Tensor n_layer_cross_k_cache;
    torch::Tensor n_layer_cross_v_cache;

    std::tie(n_layer_cross_k_cache, n_layer_cross_v_cache) =
        model_->RunEncoder(features);
#else
    std::vector<torch::Tensor> n_layer_cross_k_cache_list;
    std::vector<torch::Tensor> n_layer_cross_v_cache_list;

    for (int32_t i = 0; i != n; ++i) {
      auto features = ss[i]->GetFeatures();
      features = PadOrTrimFeatures(features).to(device).t().unsqueeze(0);

      torch::Tensor n_layer_cross_k_cache;
      torch::Tensor n_layer_cross_v_cache;

      std::tie(n_layer_cross_k_cache, n_layer_cross_v_cache) =
          model_->RunEncoder(features);
      n_layer_cross_k_cache_list.push_back(n_layer_cross_k_cache);
      n_layer_cross_v_cache_list.push_back(n_layer_cross_v_cache);
    }

    torch::Tensor n_layer_cross_k_cache =
        torch::cat(n_layer_cross_k_cache_list, 1);
    torch::Tensor n_layer_cross_v_cache =
        torch::cat(n_layer_cross_v_cache_list, 1);
#endif

    auto meta_data = model_->GetModelMetadata();
    auto sot_sequence = meta_data.sot_sequence;
    sot_sequence.push_back(meta_data.no_timestamps);
    torch::Tensor tokens =
        torch::tensor(sot_sequence, torch::dtype(torch::kLong).device(device))
            .reshape({1, -1})
            .repeat({n, 1});

    if (meta_data.is_multilingual) {
      // sot_sequence: [sot, language, task, notimestamp]
      auto language = config_.model.whisper.language;
      if (!language.empty()) {
        if (!meta_data.lang2id.count(language)) {
          SHERPA_LOG(FATAL) << "language '" << language << " is not valid";
        }
        tokens.index_put_({"...", 1}, meta_data.lang2id.at(language));
      } else {
        if (config_.model.debug) {
          SHERPA_LOGE("Begin to detect language");
        }
        auto detected_language = model_->DetectLanguage(n_layer_cross_k_cache,
                                                        n_layer_cross_v_cache);
        tokens.index_put_({"...", 1}, detected_language);

        if (config_.model.debug) {
          detected_language = detected_language.cpu();
          auto acc = detected_language.accessor<int64_t, 1>();
          for (int32_t i = 0; i != n; ++i) {
            SHERPA_LOGE("Wave %d: detected language: %s", i,
                        meta_data.id2lang.at(acc[i]).c_str());
          }
        }
      }

      if (config_.model.whisper.task == "translate") {
        tokens.index_put_({"...", 2}, meta_data.translate);
      }
    }

    torch::Tensor logits;

    torch::Tensor n_layer_self_k_cache =
        torch::zeros({meta_data.n_text_layer, n, meta_data.n_text_ctx,
                      meta_data.n_text_state},
                     torch::dtype(torch::kFloat).device(device));

    torch::Tensor n_layer_self_v_cache =
        torch::zeros({meta_data.n_text_layer, n, meta_data.n_text_ctx,
                      meta_data.n_text_state},
                     torch::dtype(torch::kFloat).device(device));

    torch::Tensor offset =
        torch::zeros({n}, torch::dtype(torch::kInt).device(device));

    std::tie(logits, n_layer_self_k_cache, n_layer_self_v_cache) =
        model_->RunDecoder(tokens, n_layer_self_k_cache, n_layer_self_v_cache,
                           n_layer_cross_k_cache, n_layer_cross_v_cache,
                           offset);

    torch::Tensor eot = torch::tensor(
        {meta_data.eot}, torch::dtype(torch::kLong).device(device));

    torch::Tensor results =
        torch::full({n, meta_data.n_text_ctx}, meta_data.eot,
                    torch::dtype(torch::kLong).device(device));

    torch::Tensor num_decoded_tokens =
        torch::zeros({n}, torch::dtype(torch::kLong).device(device));

    torch::Tensor new2old =
        torch::arange(n, torch::dtype(torch::kLong).device(device));

    for (int32_t i = 0; i < meta_data.n_text_ctx; ++i) {
      tokens = logits.slice(1, -1).argmax(-1);
      torch::Tensor eot_indexes = (tokens.squeeze() == eot).nonzero().squeeze();

      if (eot_indexes.numel()) {
        num_decoded_tokens.index_put_(
            {"...", new2old.index_select(0, eot_indexes)}, i);

        if (eot_indexes.numel() == tokens.size(0)) {
          break;
        }

        torch::Tensor non_eot_indexes =
            (tokens.squeeze() != eot).nonzero().squeeze();

        tokens = tokens.index_select(0, non_eot_indexes);

        offset = offset.index_select(0, non_eot_indexes);
        new2old = new2old.index_select(0, non_eot_indexes);
        n_layer_cross_k_cache =
            n_layer_cross_k_cache.index_select(1, non_eot_indexes);
        n_layer_cross_v_cache =
            n_layer_cross_v_cache.index_select(1, non_eot_indexes);
        n_layer_self_k_cache =
            n_layer_self_k_cache.index_select(1, non_eot_indexes);
        n_layer_self_v_cache =
            n_layer_self_v_cache.index_select(1, non_eot_indexes);
      }

      results.index_put_({new2old, i}, tokens.squeeze());
      offset.add_(logits.size(1));

      std::tie(logits, n_layer_self_k_cache, n_layer_self_v_cache) =
          model_->RunDecoder(tokens, n_layer_self_k_cache, n_layer_self_v_cache,
                             n_layer_cross_k_cache, n_layer_cross_v_cache,
                             offset);
    }
    num_decoded_tokens = num_decoded_tokens.cpu();
    auto acc = num_decoded_tokens.accessor<int64_t, 1>();
    results = results.cpu();
    auto p = results.data_ptr<int64_t>();
    for (int32_t i = 0; i != n; ++i) {
      auto token_ids = std::vector<int32_t>{p + i * results.size(1),
                                            p + i * results.size(1) + acc[i]};

      ss[i]->SetResult(Convert(token_ids, symbol_table_));
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
        sot_sequence[1] =
            model_->DetectLanguage(n_layer_cross_k_cache, n_layer_cross_v_cache)
                .item()
                .toInt();
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
