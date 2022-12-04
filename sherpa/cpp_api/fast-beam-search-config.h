// sherpa/cpp_api/fast-beam-search-config.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CPP_API_FAST_BEAM_SEARCH_CONFIG_H_
#define SHERPA_CPP_API_FAST_BEAM_SEARCH_CONFIG_H_

#include <string>

#include "sherpa/cpp_api/parse-options.h"

namespace sherpa {

// For transducer decoding with a graph
struct FastBeamSearchConfig {
  // If not empty, it is the filename of LG.pt
  // If empty, we use a trivial graph in decoding.
  std::string lg;

  // If lg is not empty, lg.scores is scaled by this value
  float ngram_lm_scale = 0.01;

  // A floating point value to calculate the cutoff score during beam
  // search (i.e., `cutoff = max-score - beam`), which is the same as the
  // `beam` in Kaldi.
  float beam = 20.0;
  int32_t max_states = 64;
  int32_t max_contexts = 8;
  bool allow_partial = false;

  void Register(ParseOptions *po);

  void Validate() const;
  std::string ToString() const;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_FAST_BEAM_SEARCH_CONFIG_H_
