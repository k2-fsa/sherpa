// sherpa/csrc/offline-transducer-greedy-search.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_H_
#define SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_H_

#include <vector>

#include "sherpa/csrc/offline-transducer-model.h"

namespace sherpa {

struct OfflineTransducerGreedySearchResults {
  /// The decoded token IDs
  std::vector<int32_t> tokens;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;
};

class OfflineTransducerGreedySearch {
 public:
  explicit OfflineTransducerGreedySearch(OfflineTransducerModel *model)
      : model_(model) {}

  /** Run greedy search given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   * @param encoder_out_length A 1-D tensor of shape (N,) containing number
   *                           of valid frames in encoder_out before padding.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  std::vector<OfflineTransducerGreedySearchResults> Decode(
      torch::Tensor encoder_out, torch::Tensor encoder_out_length);

 private:
  OfflineTransducerModel *model_;  // Not owned
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_TRANSDUCER_GREEDY_SEARCH_H_
