// sherpa/csrc/online-transducer-modified-beam-search-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/csrc/online-transducer-modified-beam-search-decoder.h"

#include <algorithm>
#include <utility>

#include "k2/torch_api.h"

namespace sherpa {

static torch::Tensor FloorDivide(torch::Tensor a, int32_t b) {
#if SHERPA_TORCH_VERSION_MAJOR > 1 || \
    (SHERPA_TORCH_VERSION_MAJOR == 1 && SHERPA_TORCH_VERSION_MINOR > 7)
  return torch::div(a, b, /*rounding_mode*/ "trunc");
#else
  return torch::floor_divide(a, b);
#endif
}

static torch::Tensor BuildDecoderInput(const std::vector<Hypothesis> &hyps,
                                       int32_t context_size) {
  int32_t num_hyps = hyps.size();
  torch::Tensor decoder_input =
      torch::empty({num_hyps, context_size},
                   torch::dtype(torch::kLong)
                       .memory_format(torch::MemoryFormat::Contiguous));

  int64_t *p = decoder_input.data_ptr<int64_t>();
  for (const auto &h : hyps) {
    auto start = h.ys.end() - context_size;
    auto end = h.ys.end();

    std::copy(start, end, p);
    p += context_size;
  }

  return decoder_input;
}

/** Return a ragged shape with axes [utt][num_hyps].
 *
 * @param hyps hyps.size() == batch_size. Each entry contains the active
 *              hypotheses of an utterance.
 * @return Return a ragged shape with 2 axes [utt][num_hyps]. Note that the
 *         shape is on CPU.
 */
static k2::RaggedShapePtr GetHypsShape(const std::vector<Hypotheses> &hyps) {
  int32_t num_utt = hyps.size();
  torch::Tensor row_splits = torch::empty(
      {num_utt + 1},
      torch::dtype(torch::kInt).memory_format(torch::MemoryFormat::Contiguous));
  auto row_splits_acc = row_splits.accessor<int32_t, 1>();
  for (int32_t i = 0; i != num_utt; ++i) {
    row_splits_acc[i] = hyps[i].Size();
  }

  k2::ExclusiveSum(row_splits, &row_splits);

  return k2::RaggedShape2(row_splits, torch::Tensor(), row_splits_acc[num_utt]);
}

OnlineTransducerDecoderResult
OnlineTransducerModifiedBeamSearchDecoder::GetEmptyResult() {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
                         //
  std::vector<int32_t> blanks(context_size, -1);
  blanks.back() = blank_id;

  Hypotheses blank_hyp({{blanks, 0}});

  OnlineTransducerDecoderResult r;
  r.hyps = std::move(blank_hyp);

  return r;
}

void OnlineTransducerModifiedBeamSearchDecoder::StripLeadingBlanks(
    OnlineTransducerDecoderResult *r) {
  int32_t context_size = model_->ContextSize();
  auto hyp = r->hyps.GetMostProbable(true);

  auto start = hyp.ys.begin() + context_size;
  auto end = hyp.ys.end();

  r->tokens = std::vector<int32_t>(start, end);
  r->timestamps = std::move(hyp.timestamps);
  r->num_trailing_blanks = hyp.num_trailing_blanks;
}

void OnlineTransducerModifiedBeamSearchDecoder::FinalizeResult(
    OnlineStream *s, OnlineTransducerDecoderResult *r) {
  if (nullptr != s->GetContextGraph()) {
    for (auto iter = r->hyps.begin(); iter != r->hyps.end(); ++iter) {
      auto context_res =
          s->GetContextGraph()->Finalize(iter->second.context_state);
      iter->second.log_prob += context_res.first;
      iter->second.context_state = context_res.second;
    }
  }
}

void OnlineTransducerModifiedBeamSearchDecoder::Decode(
    torch::Tensor encoder_out,
    std::vector<OnlineTransducerDecoderResult> *results) {
  Decode(encoder_out, nullptr, 0, results);
}

void OnlineTransducerModifiedBeamSearchDecoder::Decode(
    torch::Tensor encoder_out, OnlineStream **ss, int32_t num_streams,
    std::vector<OnlineTransducerDecoderResult> *results) {
  TORCH_CHECK(encoder_out.dim() == 3, encoder_out.dim(), " vs ", 3);

  TORCH_CHECK(encoder_out.size(0) == static_cast<int32_t>(results->size()),
              encoder_out.size(0), " vs ", results->size());

  auto device = model_->Device();
  int32_t blank_id = 0;  // always 0
  int32_t context_size = model_->ContextSize();

  int32_t N = encoder_out.size(0);
  int32_t T = encoder_out.size(1);

  if (ss) {
    SHERPA_CHECK_EQ(N, num_streams);
  }

  std::vector<Hypotheses> cur;
  cur.reserve(N);

  for (auto &r : *results) {
    cur.push_back(std::move(r.hyps));
  }

  std::vector<Hypothesis> prev;

  for (int32_t t = 0; t != T; ++t) {
    auto cur_encoder_out = encoder_out.index({torch::indexing::Slice(), t});
    // cur_encoder_out has shape (N, joiner_dim)

    // Due to merging paths with identical token sequences,
    // not all utterances have "num_active_paths" paths.
    auto hyps_shape = GetHypsShape(cur);
    int32_t num_hyps = k2::TotSize(hyps_shape, 1);

    prev.clear();
    prev.reserve(num_hyps);
    for (auto &hyps : cur) {
      for (auto &h : hyps) {
        prev.push_back(std::move(h.second));
      }
    }
    cur.clear();
    cur.reserve(N);

    auto ys_log_probs = torch::empty({num_hyps, 1}, torch::kFloat);

    auto ys_log_probs_acc = ys_log_probs.accessor<float, 2>();
    for (int32_t k = 0; k != num_hyps; ++k) {
      ys_log_probs_acc[k][0] = prev[k].log_prob;
    }

    auto decoder_input = BuildDecoderInput(prev, context_size).to(device);
    auto decoder_out = model_->RunDecoder(decoder_input).squeeze(1);
    // decoder_out is of shape (num_hyps, joiner_dim)

    auto index = k2::RowIds(hyps_shape, 1).to(torch::kLong).to(device);
    cur_encoder_out = cur_encoder_out.index_select(/*dim*/ 0, /*index*/ index);
    // cur_encoder_out is of shape (num_hyps, joiner_dim)

    auto logits = model_->RunJoiner(cur_encoder_out, decoder_out);
    // logits has shape (num_hyps, vocab_size)

    auto log_probs = (logits / temperature_).log_softmax(-1).cpu();

    log_probs.add_(ys_log_probs);

    int32_t vocab_size = log_probs.size(1);
    log_probs = log_probs.reshape(-1);
    auto row_splits = k2::RowSplits(hyps_shape, 1);
    auto row_splits_acc = row_splits.accessor<int32_t, 1>();

    for (int32_t k = 0; k != N; ++k) {
      int32_t frame_offset = (*results)[k].frame_offset;

      int32_t start = row_splits_acc[k];
      int32_t end = row_splits_acc[k + 1];

      torch::Tensor values, indexes;
      std::tie(values, indexes) =
          log_probs.slice(/*dim*/ 0, start * vocab_size, end * vocab_size)
              .topk(/*k*/ num_active_paths_, /*dim*/ 0,
                    /*largest*/ true, /*sorted*/ true);

      auto topk_hyp_indexes = FloorDivide(indexes, vocab_size);
      auto topk_token_indexes = torch::remainder(indexes, vocab_size);

      auto values_acc = values.accessor<float, 1>();
      auto topk_hyp_indexes_acc = topk_hyp_indexes.accessor<int64_t, 1>();
      auto topk_token_indexes_acc = topk_token_indexes.accessor<int64_t, 1>();

      Hypotheses hyps;
      for (int32_t j = 0; j != values.numel(); ++j) {
        int32_t hyp_idx = topk_hyp_indexes_acc[j];
        Hypothesis new_hyp = prev[start + hyp_idx];  // note: hyp_idx is 0 based

        int32_t new_token = topk_token_indexes_acc[j];

        float context_score = 0;
        auto context_state = new_hyp.context_state;

        if (new_token != blank_id) {
          new_hyp.ys.push_back(new_token);
          new_hyp.timestamps.push_back(t + frame_offset);
          new_hyp.num_trailing_blanks = 0;
          if (ss != nullptr && ss[k]->GetContextGraph() != nullptr) {
            auto context_res = ss[k]->GetContextGraph()->ForwardOneStep(
                context_state, new_token);
            context_score = context_res.first;
            new_hyp.context_state = context_res.second;
          }
        } else {
          new_hyp.num_trailing_blanks += 1;
        }

        // We already added log_prob of the path to log_probs before, so
        // we use values_acc[j] here directly.
        new_hyp.log_prob = values_acc[j] + context_score;
        hyps.Add(std::move(new_hyp));
      }
      cur.push_back(std::move(hyps));
    }  // for (int32_t k = 0; k != N; ++k)
  }    // for (int32_t t = 0; t != T; ++t)

  for (int32_t i = 0; i != N; ++i) {
    (*results)[i].hyps = std::move(cur[i]);
    (*results)[i].frame_offset += T;
  }
}

}  // namespace sherpa
