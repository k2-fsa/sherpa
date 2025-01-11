// sherpa/csrc/offline-sense-voice-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_SENSE_VOICE_MODEL_META_DATA_H_
#define SHERPA_CSRC_OFFLINE_SENSE_VOICE_MODEL_META_DATA_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "torch/script.h"

namespace sherpa {

struct OfflineSenseVoiceModelMetaData {
  // ID for using inverse text normalization
  int32_t with_itn_id;

  // ID for not using inverse text normalization
  int32_t without_itn_id;

  int32_t window_size;   // lfr_m
  int32_t window_shift;  // lfr_n
  int32_t vocab_size;

  int32_t subsampling_factor = 1;

  // Usually 0 for SenseVoice models.
  // 0 means samples are scaled to [-32768, 32767] before they are sent to the
  // feature extractor
  int32_t normalize_samples = 0;

  int32_t blank_id = 0;

  // possible values:
  // zh, en, ja, ko, yue, auto
  // where
  //  zh is Chinese (Mandarin)
  //  en is English
  //  ja is Japanese
  //  ko is Korean
  //  yue is Cantonese
  //  auto is to let the model recognize the language
  std::unordered_map<std::string, int32_t> lang2id;

  torch::Tensor neg_mean;    // 2-d float32, (1, feat_dim)
  torch::Tensor inv_stddev;  // 2-d float32, (1, feat_dim)

  std::string ToString() const;
};
}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_SENSE_VOICE_MODEL_META_DATA_H_
