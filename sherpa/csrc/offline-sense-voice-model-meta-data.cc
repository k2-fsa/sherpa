// sherpa/csrc/offline-sense-voice-model-meta-data.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa/csrc/offline-sense-voice-model-meta-data.h"

#include <sstream>
namespace sherpa {

std::string OfflineSenseVoiceModelMetaData::ToString() const {
  std::ostringstream os;
  os << "----SenseVoice metadata----\n";
  os << " with_itn_id: " << with_itn_id << "\n";
  os << " without_itn_id: " << without_itn_id << "\n";
  os << " window_size: " << window_size << "\n";
  os << " window_shift: " << window_shift << "\n";
  os << " vocab_size: " << vocab_size << "\n";
  os << " subsampling_factor: " << subsampling_factor << "\n";
  os << " normalize_samples: " << normalize_samples << "\n";
  os << " blank_id: " << blank_id << "\n";
  for (const auto &p : lang2id) {
    os << " " << p.first << ": " << p.second << "\n";
  }
  os << " neg_mean (" << neg_mean.size(1) << "): ";

  auto p = neg_mean.data_ptr<float>();
  for (int32_t i = 0; i < 10; ++i) {
    os << p[i] << ", ";
  }
  os << "\n";

  os << " inv_stddev (" << inv_stddev.size(1) << "): ";
  p = inv_stddev.data_ptr<float>();
  for (int32_t i = 0; i < 10; ++i) {
    os << p[i] << ", ";
  }
  os << "\n";

  return os.str();
}

}  // namespace sherpa
