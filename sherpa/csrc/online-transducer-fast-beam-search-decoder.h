// sherpa/csrc/online-transducer-fast-beam-search-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_ONLINE_TRANSDUCER_FAST_BEAM_SEARCH_DECODER_H_
#define SHERPA_CSRC_ONLINE_TRANSDUCER_FAST_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/csrc/online-transducer-decoder.h"
#include "sherpa/csrc/online-transducer-model.h"

namespace sherpa {

struct FastBeamSearchConfig {
  // If not empty, it is the filename of LG.pt
  // If empty, we use a trivial graph in decoding.
  std::string lg;

  // If lg is not empty, lg.scores is scaled by this value
  float ngram_lm_scale;

  // A floating point value to calculate the cutoff score during beam
  // search (i.e., `cutoff = max-score - beam`), which is the same as the
  //`beam` in Kaldi.
  float beam = 20.0;
  int32_t max_states = 64;
  int32_t max_contexts = 8;
  bool allow_partial = false;
};

class OnlineTransducerFastBeamSearchDecoder : public OnlineTransducerDecoder {
 public:
  /**
   * @param config
   * @param vocab_size Used only when config.lg is empty.
   */
  OnlineTransducerFastBeamSearchDecoder(OnlineTransducerModel *model,
                                        const FastBeamSearchConfig &config,
                                        int32_t vocab_size);

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
