// sherpa/csrc/online-transducer-greedy-search-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/csrc/online-transducer-decoder.h"
#include "sherpa/csrc/online-transducer-model.h"

namespace sherpa {

class OnlineTransducerGreedySearchDecoder : public OnlineTransducerDecoder {
 public:
  explicit OnlineTransducerGreedySearchDecoder(OnlineTransducerModel *model)
      : model_(model) {}

  OnlineTransducerDecoderResult GetEmptyResult() override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) override;

  void Decode(torch::Tensor encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

 private:
  OnlineTransducerModel *model_;  // Not owned
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
