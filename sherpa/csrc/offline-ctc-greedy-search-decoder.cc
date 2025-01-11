// sherpa/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/offline-ctc-greedy-search-decoder.h"

#include <utility>

#include "sherpa/cpp_api/macros.h"

namespace sherpa {

std::vector<OfflineCtcDecoderResult> OfflineCtcGreedySearchDecoder::Decode(
    torch::Tensor logits, torch::Tensor logits_len,
    int32_t subsampling_factor /*= 1*/) {
  InferenceMode no_grad;

  int32_t batch_size = logits.size(0);

  torch::Tensor indexes = logits.argmax(-1);

  logits_len = logits_len.to(torch::kInt).cpu();

  auto p_len = logits_len.accessor<int32_t, 1>();

  std::vector<OfflineCtcDecoderResult> results(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    torch::Tensor this_indexes = indexes.index({i}).slice(0, 0, p_len[i]);

    this_indexes = std::get<0>(torch::unique_consecutive(this_indexes));

    // assume that the blank id is 0
    torch::Tensor non_zero_indexes = this_indexes.nonzero().squeeze();
    torch::Tensor tokens =
        this_indexes.index_select(0, non_zero_indexes).cpu().to(torch::kInt);

    results[i].tokens = {tokens.data_ptr<int32_t>(),
                         tokens.data_ptr<int32_t>() + tokens.numel()};
  }

  return results;
}

}  // namespace sherpa
