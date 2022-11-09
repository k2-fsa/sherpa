// sherpa/csrc/offline-ctc-one-best-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_CTC_ONE_BEST_DECODER_H_
#define SHERPA_CSRC_OFFLINE_CTC_ONE_BEST_DECODER_H_

#include <vector>

#include "k2/torch_api.h"
#include "sherpa/cpp_api/offline-recognizer.h"
#include "sherpa/csrc/offline-ctc-decoder.h"

namespace sherpa {

class OfflineCtcOneBestDecoder : public OfflineCtcDecoder {
 public:
  /**
   * @param vocab_size Output dimension of the model.
   */
  OfflineCtcOneBestDecoder(const OfflineCtcDecoderConfig config,
                           torch::Device device);

  std::vector<OfflineCtcDecoderResult> Decode(
      torch::Tensor log_prob, torch::Tensor log_prob_len,
      int32_t subsampling_factor = 1) override;

 private:
  OfflineCtcDecoderConfig config_;
  k2::FsaClassPtr decoding_graph_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_CTC_ONE_BEST_DECODER_H_
