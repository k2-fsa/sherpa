// sherpa/cpp_api/offline-recognizer-whisper-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/symbol-table.h"

namespace sherpa {

class OfflineRecognizerWhisperImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWhisperImpl(const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config.model.tokens),
        fbank_(config.feat_config.fbank_opts) {
    SHERPA_LOG(INFO) << "called";
  }

  std::unique_ptr<OfflineStream> CreateStream() override {
    return std::make_unique<OfflineStream>(&fbank_, config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) override {
    InferenceMode no_grad;
  }

 private:
  void WarmUp() {
    SHERPA_LOG(INFO) << "WarmUp begins";

    SHERPA_LOG(INFO) << "WarmUp ended";
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  kaldifeat::Fbank fbank_;
};
}  // namespace sherpa
#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
