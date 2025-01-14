// sherpa/csrc/offline-whisper-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_WHISPER_MODEL_META_DATA_H_
#define SHERPA_CSRC_OFFLINE_WHISPER_MODEL_META_DATA_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "torch/script.h"

namespace sherpa {

struct OfflineWhisperModelMetaData {
  int32_t n_mels;
  int32_t n_audio_ctx;
  int32_t n_audio_state;
  int32_t n_audio_head;
  int32_t n_audio_layer;
  int32_t n_vocab;
  int32_t n_text_ctx;
  int32_t n_text_state;
  int32_t n_text_head;
  int32_t n_text_layer;
  int32_t sot;
  int32_t sot_index;
  int32_t eot;
  int32_t blank_id;
  int32_t is_multilingual;
  int32_t no_speech;
  int32_t non_speech_tokens;
  int32_t transcribe;
  int32_t translate;
  int32_t sot_prev;
  int32_t sot_lm;
  int32_t no_timestamps;

  std::string comment;
  std::vector<int64_t> sot_sequence;
  std::unordered_map<std::string, int32_t> lang2id;

  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_WHISPER_MODEL_META_DATA_H_
