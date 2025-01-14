// sherpa/csrc/offline-whisper-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/offline-whisper-model-meta-data.h"

#include <algorithm>
#include <sstream>
#include <string>

namespace sherpa {

std::string OfflineWhisperModelMetaData::ToString() const {
  std::ostringstream os;

  os << "----------whisper meta data----------\n";

  os << " comment: " << comment << "\n";
  os << " n_mels: " << n_mels << "\n";
  os << " n_audio_ctx: " << n_audio_ctx << "\n";
  os << " n_audio_state: " << n_audio_state << "\n";
  os << " n_audio_head: " << n_audio_head << "\n";
  os << " n_audio_layer: " << n_audio_layer << "\n";
  os << " n_vocab: " << n_vocab << "\n";
  os << " n_text_ctx: " << n_text_ctx << "\n";
  os << " n_text_state: " << n_text_state << "\n";
  os << " n_text_head: " << n_text_head << "\n";
  os << " n_text_layer: " << n_text_layer << "\n";
  os << " sot: " << sot << "\n";
  os << " sot_index: " << sot_index << "\n";
  os << " eot: " << eot << "\n";
  os << " blank_id: " << blank_id << "\n";
  os << " is_multilingual: " << is_multilingual << "\n";
  os << " no_speech: " << no_speech << "\n";
  os << " non_speech_tokens: " << non_speech_tokens << "\n";
  os << " transcribe: " << transcribe << "\n";
  os << " translate: " << translate << "\n";
  os << " sot_prev: " << sot_prev << "\n";
  os << " sot_lm: " << sot_lm << "\n";
  os << " no_timestamps: " << no_timestamps << "\n";
  os << " sot_sequence:";
  for (auto i : sot_sequence) {
    os << " " << i;
  }
  os << "\n";

  std::vector<std::string> langs;
  langs.reserve(lang2id.size());
  for (const auto &p : lang2id) {
    langs.push_back(p.first);
  }
  std::sort(langs.begin(), langs.end());

  os << " lang2id: (" << lang2id.size() << ")" << "\n    ";
  int32_t k = 0;
  for (const auto &lang : langs) {
    os << lang << " -> " << lang2id.at(lang) << ", ";
    k += 1;
    if (k % 10 == 0) {
      os << "\n    ";
    }
  }
  os << "\n";

  return os.str();
}

}  // namespace sherpa
