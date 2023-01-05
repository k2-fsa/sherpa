// sherpa/csrc/offline-ctc-one-best-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/offline-ctc-one-best-decoder.h"

#include <utility>

#include "sherpa/csrc/log.h"

namespace sherpa {

OfflineCtcOneBestDecoder::OfflineCtcOneBestDecoder(
    const OfflineCtcDecoderConfig &config, torch::Device device,
    int32_t vocab_size)
    : config_(config), vocab_size_(vocab_size) {
  if (config.hlg.empty()) {
    // Use CTC topo since no HLG is provided
    SHERPA_CHECK_GT(vocab_size, 1);

    decoding_graph_ = k2::GetCtcTopo(vocab_size - 1, config.modified, device);
  } else {
    decoding_graph_ = k2::LoadFsaClass(config.hlg, device);

    k2::ScaleTensorAttribute(decoding_graph_, config.lm_scale, "scores");
  }
}

std::vector<OfflineCtcDecoderResult> OfflineCtcOneBestDecoder::Decode(
    torch::Tensor log_prob, torch::Tensor log_prob_len,
    int32_t subsampling_factor /*= 1*/) {
  if (vocab_size_ > 0) {
    SHERPA_CHECK_EQ(log_prob.size(2), vocab_size_);
  }

  torch::NoGradGuard no_grad;

  auto lattice = k2::GetLattice(log_prob, log_prob_len.cpu(), decoding_graph_,
                                config_.search_beam, config_.output_beam,
                                config_.min_active_states,
                                config_.max_active_states, subsampling_factor);

  lattice = k2::ShortestPath(lattice);
  std::vector<OfflineCtcDecoderResult> results(log_prob.size(0));

  // Get tokens and timestamps from the lattice
  auto labels = k2::GetTensorAttr(lattice, "labels").cpu().contiguous();
  auto acc = labels.accessor<int32_t, 1>();

  OfflineCtcDecoderResult *p = results.data();

  for (int32_t i = 0, t = 0; i != labels.numel(); ++i) {
    int32_t token = acc[i];

    if (token == -1) {
      // end of this utterance.
      t = 0;
      ++p;

      continue;
    }

    if (token == 0) {
      ++t;
      continue;
    }
    if (t != 0 && !p->tokens.empty() && token == p->tokens.back()) {
      // This is a repeat, skip it.
      ++t;
      continue;
    }

    p->tokens.push_back(token);
    p->timestamps.push_back(t);
    ++t;
  }  // for (int32_t i = 0, t = 0; i != labels.numel(); ++i)

  return results;
}

}  // namespace sherpa
