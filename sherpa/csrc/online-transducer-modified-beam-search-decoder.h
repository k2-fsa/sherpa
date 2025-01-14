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
      OnlineTransducerModel *model, int32_t num_active_paths, float temperature)
      : model_(model),
        num_active_paths_(num_active_paths),
        temperature_(temperature) {}

  OnlineTransducerDecoderResult GetEmptyResult() override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) override;

  void FinalizeResult(OnlineStream *s,
                      OnlineTransducerDecoderResult *r) override;

  void Decode(torch::Tensor encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

  void Decode(torch::Tensor encoder_out, OnlineStream **ss, int32_t num_streams,
              std::vector<OnlineTransducerDecoderResult> *result) override;

 private:
  OnlineTransducerModel *model_;  // Not owned
  int32_t num_active_paths_;
  float temperature_ = 1.0;
};

}  // namespace sherpa
#endif  // SHERPA_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
