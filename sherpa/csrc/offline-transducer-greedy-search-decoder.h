// sherpa/csrc/offline-transducer-greedy-search-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/cpp_api/offline-stream.h"
#include "sherpa/csrc/offline-transducer-decoder.h"
#include "sherpa/csrc/offline-transducer-model.h"

namespace sherpa {

class OfflineTransducerGreedySearchDecoder : public OfflineTransducerDecoder {
 public:
  explicit OfflineTransducerGreedySearchDecoder(OfflineTransducerModel *model)
      : model_(model) {}

  /** Run greedy search given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   * @param encoder_out_length A 1-D tensor of shape (N,) containing number
   *                           of valid frames in encoder_out before padding.
   * @param ss Pointer to an array of streams.
   * @param n  Size of the input array.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  std::vector<OfflineTransducerDecoderResult> Decode(
      torch::Tensor encoder_out, torch::Tensor encoder_out_length,
      OfflineStream **ss = nullptr, int32_t n = 0) override;

 private:
  OfflineTransducerModel *model_;  // Not owned
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_H_
