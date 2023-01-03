// sherpa/csrc/offline-transducer-greedy-search-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/csrc/offline-transducer-decoder.h"
#include "sherpa/csrc/offline-transducer-model.h"

namespace sherpa {

class OfflineTransducerGreedySearchDecoder : public OfflineTransducerDecoder {
 public:
  explicit OfflineTransducerGreedySearchDecoder(OfflineTransducerModel *model)
      : model_(model) {}

  std::vector<OfflineTransducerDecoderResult> Decode(
      torch::Tensor encoder_out, torch::Tensor encoder_out_length) override;

 private:
  OfflineTransducerModel *model_;  // Not owned
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
