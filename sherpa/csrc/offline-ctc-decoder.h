// sherpa/csrc/offline-ctc-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_CTC_DECODER_H_
#define SHERPA_CSRC_OFFLINE_CTC_DECODER_H_

#include <vector>

#include "sherpa/cpp_api/parse-options.h"
#include "torch/script.h"

namespace sherpa {

struct OfflineCtcDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;
};

class OfflineCtcDecoder {
 public:
  virtual ~OfflineCtcDecoder() = default;

  /** Run CTC decoder given the output from the encoder model.
   *
   * @param log_prob A 3-D tensor of shape (N, T, vocab_size)
   * @param log_prob_len A 1-D tensor of shape (N,) containing number
   *                     of valid frames in encoder_out before padding.
   * @param subsampling_factor Subsampling factor of the model.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineCtcDecoderResult> Decode(
      torch::Tensor log_prob, torch::Tensor log_prob_len,
      int32_t subsampling_factor = 1) = 0;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_CTC_DECODER_H_
