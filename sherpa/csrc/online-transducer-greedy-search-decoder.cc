// sherpa/csrc/online-transducer-greedy-search-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/online-transducer-greedy-search-decoder.h"

#include <algorithm>
#include <vector>

namespace sherpa {

static void BuildDecoderInput(
    const std::vector<OnlineTransducerDecoderResult> &r,
    torch::Tensor *decoder_input) {
  int32_t batch_size = decoder_input->size(0);
  int32_t context_size = decoder_input->size(1);
  int64_t *p = decoder_input->data_ptr<int64_t>();
  for (int32_t i = 0; i != batch_size; ++i) {
    auto start = r[i].tokens.end() - context_size;
    auto end = r[i].tokens.end();
    std::copy(start, end, p);
    p += context_size;
  }
}

OnlineTransducerDecoderResult
OnlineTransducerGreedySearchDecoder::GetEmptyResult() {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  OnlineTransducerDecoderResult r;
  r.tokens.resize(context_size, -1);
  r.tokens.back() = blank_id;

  return r;
}

void OnlineTransducerGreedySearchDecoder::StripLeadingBlanks(
    OnlineTransducerDecoderResult *r) {
  int32_t context_size = model_->ContextSize();

  auto start = r->tokens.begin() + context_size;
  auto end = r->tokens.end();

  r->tokens = std::vector<int32_t>(start, end);
}

void OnlineTransducerGreedySearchDecoder::Decode(
    torch::Tensor encoder_out,
    std::vector<OnlineTransducerDecoderResult> *results) {
  TORCH_CHECK(encoder_out.dim() == 3, encoder_out.dim(), " vs ", 3);

  TORCH_CHECK(encoder_out.size(0) == static_cast<int32_t>(results->size()),
              encoder_out.size(0), " vs ", results->size());

  auto device = model_->Device();
  int32_t blank_id = 0;  // always 0
  int32_t context_size = model_->ContextSize();

  int32_t N = encoder_out.size(0);
  int32_t T = encoder_out.size(1);

  auto decoder_input = torch::empty(
      {N, context_size}, torch::dtype(torch::kLong)
                             .memory_format(torch::MemoryFormat::Contiguous));
  BuildDecoderInput(*results, &decoder_input);

  auto decoder_out = model_->RunDecoder(decoder_input.to(device)).squeeze(1);
  // decoder_out has shape (N, joiner_dim)

  for (int32_t t = 0; t != T; ++t) {
    auto cur_encoder_out = encoder_out.index({torch::indexing::Slice(), t});
    // cur_encoder_out has shape (N, joiner_dim)

    auto logits = model_->RunJoiner(cur_encoder_out, decoder_out);
    // logits has shape (N, vocab_size)

    auto max_indices = logits.argmax(/*dim*/ -1).cpu();
    auto max_indices_accessor = max_indices.accessor<int64_t, 1>();
    bool emitted = false;
    for (int32_t n = 0; n != N; ++n) {
      auto index = max_indices_accessor[n];
      auto &r = (*results)[n];
      if (index != blank_id) {
        emitted = true;

        r.tokens.push_back(index);
        r.timestamps.push_back(t + r.frame_offset);
        r.num_trailing_blanks = 0;
      } else {
        ++r.num_trailing_blanks;
      }
    }

    if (emitted) {
      BuildDecoderInput(*results, &decoder_input);
      decoder_out = model_->RunDecoder(decoder_input.to(device)).squeeze(1);
      // decoder_out has shape (N, joiner_dim)
    }
  }  // for (int32_t t = 0; t != T; ++t)

  // Update frame_offset
  for (auto &r : *results) {
    r.frame_offset += T;
  }
}

}  // namespace sherpa
