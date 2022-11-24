// sherpa/csrc/online-transducer-fast-beam-search-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_TRANSDUCER_FAST_BEAM_SEARCH_DECODER_H_
#define SHERPA_CSRC_ONLINE_TRANSDUCER_FAST_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/cpp_api/online-recognizer.h"
#include "sherpa/csrc/online-transducer-decoder.h"
#include "sherpa/csrc/online-transducer-model.h"

namespace sherpa {

class OnlineTransducerFastBeamSearchDecoder : public OnlineTransducerDecoder {
 public:
  /**
   * @param config
   */
  OnlineTransducerFastBeamSearchDecoder(OnlineTransducerModel *model,
                                        const FastBeamSearchConfig &config);

  /* Return an empty result. */
  OnlineTransducerDecoderResult GetEmptyResult() override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) override;

  void Decode(torch::Tensor encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

 private:
  OnlineTransducerModel *model_;  // Not owned
  k2::FsaClassPtr decoding_graph_;

  FastBeamSearchConfig config_;
  int32_t vocab_size_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_TRANSDUCER_FAST_BEAM_SEARCH_DECODER_H_
