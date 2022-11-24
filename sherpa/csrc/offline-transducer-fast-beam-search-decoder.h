// sherpa/csrc/offline-transducer-fast-beam-search-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_TRANSDUCER_FAST_BEAM_SEARCH_DECODER_H_
#define SHERPA_CSRC_OFFLINE_TRANSDUCER_FAST_BEAM_SEARCH_DECODER_H_
#include <vector>

#include "k2/torch_api.h"
#include "sherpa/cpp_api/fast-beam-search-config.h"
#include "sherpa/csrc/offline-transducer-decoder.h"
#include "sherpa/csrc/offline-transducer-model.h"

namespace sherpa {

class OfflineTransducerFastBeamSearchDecoder : public OfflineTransducerDecoder {
 public:
  OfflineTransducerFastBeamSearchDecoder(OfflineTransducerModel *model,
                                         const FastBeamSearchConfig &config);

  /** Run fast_beam_search given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   * @param encoder_out_length A 1-D tensor of shape (N,) containing number
   *                           of valid frames in encoder_out before padding.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  std::vector<OfflineTransducerDecoderResult> Decode(
      torch::Tensor encoder_out, torch::Tensor encoder_out_length) override;

 private:
  OfflineTransducerModel *model_;  // Not owned
  k2::FsaClassPtr decoding_graph_;

  FastBeamSearchConfig config_;
  int32_t vocab_size_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_TRANSDUCER_FAST_BEAM_SEARCH_DECODER_H_
