// sherpa/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_
#define SHERPA_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/csrc/offline-ctc-decoder.h"

namespace sherpa {

class OfflineCtcGreedySearchDecoder : public OfflineCtcDecoder {
 public:
  std::vector<OfflineCtcDecoderResult> Decode(
      torch::Tensor logits, torch::Tensor logits_len,
      int32_t subsampling_factor = 1) override;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_
