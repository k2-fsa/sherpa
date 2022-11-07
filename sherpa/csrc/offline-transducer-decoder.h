// sherpa/csrc/offline-transducer-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_TRANSDUCER_DECODER_H_
#define SHERPA_CSRC_OFFLINE_TRANSDUCER_DECODER_H_

#include <vector>

#include "torch/script.h"

namespace sherpa {

struct OfflineTransducerDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;
};

class OfflineTransducerDecoder {
 public:
  virtual ~OfflineTransducerDecoder() = default;

  /** Run transducer beam search given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   * @param encoder_out_length A 1-D tensor of shape (N,) containing number
   *                           of valid frames in encoder_out before padding.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineTransducerDecoderResult> Decode(
      torch::Tensor encoder_out, torch::Tensor encoder_out_length) = 0;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_TRANSDUCER_DECODER_H_
