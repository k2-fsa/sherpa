// sherpa/csrc/offline-transducer-modified-beam-search-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/offline-transducer-modified-beam-search-decoder.h"

#include <algorithm>
#include <deque>
#include <utility>

#include "k2/torch_api.h"
#include "sherpa/csrc/hypothesis.h"
#include "sherpa/csrc/log.h"

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

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerModifiedBeamSearchDecoder::Decode(
    torch::Tensor encoder_out, torch::Tensor encoder_out_length,
    OfflineStream **ss /*= nullptr*/, int32_t n /*= 0*/) {
  TORCH_CHECK(encoder_out.dim() == 3, "encoder_out.dim() is ",
              encoder_out.dim(), "Expected value is 3");
  TORCH_CHECK(encoder_out.scalar_type() == torch::kFloat,
              "encoder_out.scalar_type() is ", encoder_out.scalar_type());

  TORCH_CHECK(encoder_out_length.dim() == 1, "encoder_out_length.dim() is",
              encoder_out_length.dim());
  TORCH_CHECK(encoder_out_length.scalar_type() == torch::kLong,
              "encoder_out_length.scalar_type() is ",
              encoder_out_length.scalar_type());

  TORCH_CHECK(encoder_out_length.device().is_cpu());

  torch::Device device = model_->Device();
  encoder_out = encoder_out.to(device);

  torch::nn::utils::rnn::PackedSequence packed_seq =
      torch::nn::utils::rnn::pack_padded_sequence(encoder_out,
                                                  encoder_out_length,
                                                  /*batch_first*/ true,
                                                  /*enforce_sorted*/ false);

  auto packed_encoder_out = packed_seq.data();

  int32_t blank_id = 0;
  int32_t context_size = model_->ContextSize();

  int32_t batch_size = encoder_out_length.size(0);

  if (ss != nullptr) SHERPA_CHECK_EQ(batch_size, n);

  std::vector<int32_t> blanks(context_size, -1);
  blanks.back() = blank_id;

  Hypotheses blank_hyp({{blanks, 0}});

  std::deque<Hypotheses> finalized;
  std::vector<Hypotheses> cur;
  std::vector<Hypothesis> prev;

  std::vector<ContextGraphPtr> context_graphs(batch_size, nullptr);

  auto sorted_indices = packed_seq.sorted_indices().cpu();
  auto sorted_indices_accessor = sorted_indices.accessor<int64_t, 1>();

  for (int32_t i = 0; i < batch_size; ++i) {
    const ContextState *context_state = nullptr;
    if (ss != nullptr) {
      context_graphs[i] = ss[sorted_indices_accessor[i]]->GetContextGraph();
      if (context_graphs[i] != nullptr)
        context_state = context_graphs[i]->Root();
    }
    Hypotheses blank_hyp({{blanks, 0, context_state}});
    cur.emplace_back(std::move(blank_hyp));
  }

  using torch::indexing::Slice;
  auto batch_sizes_acc = packed_seq.batch_sizes().accessor<int64_t, 1>();
  int32_t max_T = packed_seq.batch_sizes().numel();
  int32_t offset = 0;

  for (int32_t t = 0; t != max_T; ++t) {
    int32_t cur_batch_size = batch_sizes_acc[t];
    int32_t start = offset;
    int32_t end = start + cur_batch_size;
    auto cur_encoder_out = packed_encoder_out.index({Slice(start, end)});
    offset = end;

    cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // Now cur_encoder_out's shape is (cur_batch_size, 1, 1, joiner_dim)

    if (cur_batch_size < static_cast<int32_t>(cur.size())) {
      for (int32_t k = static_cast<int32_t>(cur.size()) - 1;
           k >= cur_batch_size; --k) {
        finalized.push_front(std::move(cur[k]));
      }
      cur.erase(cur.begin() + cur_batch_size, cur.end());
    }

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
    cur.reserve(cur_batch_size);

    auto ys_log_probs = torch::empty({num_hyps, 1}, torch::kFloat);

    auto ys_log_probs_acc = ys_log_probs.accessor<float, 2>();
    for (int32_t k = 0; k != static_cast<int32_t>(prev.size()); ++k) {
      ys_log_probs_acc[k][0] = prev[k].log_prob;
    }

    auto decoder_input = BuildDecoderInput(prev, context_size).to(device);

    auto decoder_out = model_->RunDecoder(decoder_input);
    // decoder_out is of shape (num_hyps, 1, joiner_dim)

    auto index = k2::RowIds(hyps_shape, 1).to(torch::kLong).to(device);

    cur_encoder_out = cur_encoder_out.index_select(/*dim*/ 0, /*index*/ index);
    // cur_encoder_out is of shape (num_hyps, 1, 1, joiner_dim)

    auto logits = model_->RunJoiner(cur_encoder_out, decoder_out.unsqueeze(1));

    // logits' shape is (num_hyps, 1, 1, vocab_size)
    logits = logits.squeeze(1).squeeze(1);
    // now logits' shape is (num_hyps, vocab_size)

    auto log_probs = (logits / temperature_).log_softmax(-1).cpu();

    log_probs.add_(ys_log_probs);

    int32_t vocab_size = log_probs.size(1);
    log_probs = log_probs.reshape(-1);
    auto row_splits = k2::RowSplits(hyps_shape, 1);
    auto row_splits_acc = row_splits.accessor<int32_t, 1>();

    for (int32_t k = 0; k != cur_batch_size; ++k) {
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
          new_hyp.timestamps.push_back(t);
          if (context_graphs[k] != nullptr) {
            auto context_res =
                context_graphs[k]->ForwardOneStep(context_state, new_token);
            context_score = context_res.first;
            new_hyp.context_state = context_res.second;
          }
        }

        // We already added log_prob of the path to log_probs before, so
        // we use values_acc[j] here directly.
        new_hyp.log_prob = values_acc[j] + context_score;
        hyps.Add(std::move(new_hyp));
      }
      cur.push_back(std::move(hyps));
    }
  }

  for (auto &h : finalized) {
    cur.push_back(std::move(h));
  }

  // Finalize context biasing matching..
  for (int32_t i = 0; i < static_cast<int32_t>(cur.size()); ++i) {
    for (auto iter = cur[i].begin(); iter != cur[i].end(); ++iter) {
      if (context_graphs[i] != nullptr) {
        auto context_res =
            context_graphs[i]->Finalize(iter->second.context_state);
        iter->second.log_prob += context_res.first;
        iter->second.context_state = context_res.second;
      }
    }
  }

  auto unsorted_indices = packed_seq.unsorted_indices().cpu();
  auto unsorted_indices_accessor = unsorted_indices.accessor<int64_t, 1>();

  std::vector<OfflineTransducerDecoderResult> ans(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    int32_t k = unsorted_indices_accessor[i];
    Hypothesis hyp = cur[k].GetMostProbable(true);
    torch::ArrayRef<int32_t> arr(hyp.ys);
    ans[i].tokens = arr.slice(context_size).vec();
    ans[i].timestamps = std::move(hyp.timestamps);
  }

  return ans;
}

}  // namespace sherpa
