// sherpa/cpp_api/fast-beam-search-config.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/cpp_api/fast-beam-search-config.h"

#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

// TODO(fangjun): Add a prefix for it
void FastBeamSearchConfig::Register(ParseOptions *po) {
  po->Register("lg", &lg,
               "Path to LG.pt. Used only for fast_beam_search "
               "in transducer decoding");

  po->Register("ngram-lm-scale", &ngram_lm_scale,
               "Scale the scores from LG.pt. Used only for fast_beam_search "
               "in transducer decoding");

  po->Register("beam", &beam, "Beam used in fast_beam_search");
}

void FastBeamSearchConfig::Validate() const {
  if (!lg.empty()) {
    AssertFileExists(lg);
  }
  SHERPA_CHECK_GE(ngram_lm_scale, 0);
  SHERPA_CHECK_GT(beam, 0);
}

}  // namespace sherpa
