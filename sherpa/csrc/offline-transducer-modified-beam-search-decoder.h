// sherpa/csrc/offline-transducer-modified-beam-search-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/csrc/offline-transducer-decoder.h"
#include "sherpa/csrc/offline-transducer-model.h"

namespace sherpa {

class OfflineTransducerModifiedBeamSearchDecoder
    : public OfflineTransducerDecoder {
 public:
  explicit OfflineTransducerModifiedBeamSearchDecoder(
      OfflineTransducerModel *model, int32_t num_active_paths)
      : model_(model), num_active_paths_(num_active_paths) {}

  /** Run greedy search given the output from the encoder model.
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

  int32_t num_active_paths_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
