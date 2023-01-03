// sherpa/csrc/online-transducer-modified-beam-search-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/csrc/online-transducer-decoder.h"
#include "sherpa/csrc/online-transducer-model.h"

namespace sherpa {

class OnlineTransducerModifiedBeamSearchDecoder
    : public OnlineTransducerDecoder {
 public:
  explicit OnlineTransducerModifiedBeamSearchDecoder(
      OnlineTransducerModel *model, int32_t num_active_paths)
      : model_(model), num_active_paths_(num_active_paths) {}

  OnlineTransducerDecoderResult GetEmptyResult() override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) override;

  void Decode(torch::Tensor encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

 private:
  OnlineTransducerModel *model_;  // Not owned
  int32_t num_active_paths_;
};

}  // namespace sherpa
#endif  // SHERPA_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
