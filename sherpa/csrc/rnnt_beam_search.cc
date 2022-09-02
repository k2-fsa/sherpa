/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sherpa/csrc/rnnt_beam_search.h"

#include <algorithm>
#include <deque>
#include <utility>

#include "k2/torch_api.h"
#include "sherpa/csrc/hypothesis.h"
#include "sherpa/csrc/rnnt_conformer_model.h"
#include "sherpa/csrc/rnnt_emformer_model.h"
#include "sherpa/csrc/rnnt_model.h"
#include "torch/all.h"

namespace sherpa {

static inline torch::Tensor FloorDivide(torch::Tensor a, int32_t b) {
#if SHERPA_TORCH_VERSION_MAJOR > 1 || \
    (SHERPA_TORCH_VERSION_MAJOR == 1 && SHERPA_TORCH_VERSION_MINOR > 7)
  return torch::div(a, b, /*rounding_mode*/ "trunc");
#else
  return torch::floor_divide(a, b);
#endif
}

/**
 * Construct the decoder input from the current hypothesis.
 *
 * @param hyps  A list-of-list of token IDs containing the current decoding
 *              results. Its length is `batch_size`
 * @param decoder_input A 2-D tensor of shape (batch_size, context_size).
 */
static void BuildDecoderInput(const std::vector<std::vector<int32_t>> &hyps,
                              torch::Tensor *decoder_input) {
  int32_t batch_size = decoder_input->size(0);
  int32_t context_size = decoder_input->size(1);
  int64_t *p = decoder_input->data_ptr<int64_t>();
  for (int32_t i = 0; i != batch_size; ++i) {
    auto start = hyps[i].end() - context_size;
    auto end = hyps[i].end();
    std::copy(start, end, p);
    p += context_size;
  }
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

std::vector<std::vector<int32_t>> GreedySearch(
    RnntModel &model,  // NOLINT
    torch::Tensor encoder_out, torch::Tensor encoder_out_length) {
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

  torch::Device device = model.Device();
  encoder_out = encoder_out.to(device);

  torch::nn::utils::rnn::PackedSequence packed_seq =
      torch::nn::utils::rnn::pack_padded_sequence(encoder_out,
                                                  encoder_out_length,
                                                  /*batch_first*/ true,
                                                  /*enforce_sorted*/ false);

  auto projected_encoder_out = model.ForwardEncoderProj(packed_seq.data());

  int32_t blank_id = model.BlankId();
  int32_t unk_id = model.UnkId();
  int32_t context_size = model.ContextSize();

  int32_t batch_size = encoder_out_length.size(0);

  std::vector<int32_t> blanks(context_size, blank_id);
  std::vector<std::vector<int32_t>> hyps(batch_size, blanks);

  auto decoder_input =
      torch::full({batch_size, context_size}, blank_id,
                  torch::dtype(torch::kLong)
                      .memory_format(torch::MemoryFormat::Contiguous));
  auto decoder_out = model.ForwardDecoder(decoder_input.to(device));
  decoder_out = model.ForwardDecoderProj(decoder_out);
  // decoder_out's shape is (batch_size, 1, joiner_dim)

  using torch::indexing::Slice;
  auto batch_sizes_accessor = packed_seq.batch_sizes().accessor<int64_t, 1>();
  int32_t num_batches = packed_seq.batch_sizes().numel();
  int32_t offset = 0;
  for (int32_t i = 0; i != num_batches; ++i) {
    int32_t cur_batch_size = batch_sizes_accessor[i];
    int32_t start = offset;
    int32_t end = start + cur_batch_size;
    auto cur_encoder_out = projected_encoder_out.index({Slice(start, end)});
    offset = end;

    cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // Now cur_encoder_out's shape is (cur_batch_size, 1, 1, joiner_dim)
    if (cur_batch_size < decoder_out.size(0)) {
      decoder_out = decoder_out.index({Slice(0, cur_batch_size)});
    }

    auto logits =
        model.ForwardJoiner(cur_encoder_out, decoder_out.unsqueeze(1));
    // logits' shape is (cur_batch_size, 1, 1, vocab_size)

    logits = logits.squeeze(1).squeeze(1);
    auto max_indices = logits.argmax(/*dim*/ -1).cpu();
    auto max_indices_accessor = max_indices.accessor<int64_t, 1>();
    bool emitted = false;
    for (int32_t k = 0; k != cur_batch_size; ++k) {
      auto index = max_indices_accessor[k];
      if (index != blank_id && index != unk_id) {
        emitted = true;
        hyps[k].push_back(index);
      }
    }

    if (emitted) {
      if (cur_batch_size < decoder_input.size(0)) {
        decoder_input = decoder_input.index({Slice(0, cur_batch_size)});
      }
      BuildDecoderInput(hyps, &decoder_input);
      decoder_out = model.ForwardDecoder(decoder_input.to(device));
      decoder_out = model.ForwardDecoderProj(decoder_out);
    }
  }

  auto unsorted_indices = packed_seq.unsorted_indices().cpu();
  auto unsorted_indices_accessor = unsorted_indices.accessor<int64_t, 1>();

  std::vector<std::vector<int32_t>> ans(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    torch::ArrayRef<int32_t> arr(hyps[unsorted_indices_accessor[i]]);
    ans[i] = arr.slice(context_size).vec();
  }

  return ans;
}

torch::Tensor StreamingGreedySearch(
    RnntModel &model,  // NOLINT
    torch::Tensor encoder_out, torch::Tensor decoder_out,
    std::vector<std::vector<int32_t>> *hyps,
    std::vector<int32_t> *num_trailing_blank_frames) {
  TORCH_CHECK(encoder_out.dim() == 3, encoder_out.dim(), " vs ", 3);
  TORCH_CHECK(decoder_out.dim() == 2, decoder_out.dim(), " vs ", 2);

  TORCH_CHECK(encoder_out.size(0) == decoder_out.size(0), encoder_out.size(0),
              " vs ", decoder_out.size(0));

  TORCH_CHECK(encoder_out.size(0) == hyps->size(), encoder_out.size(0), " vs ",
              hyps->size());

  TORCH_CHECK(hyps->size() == num_trailing_blank_frames->size(), hyps->size(),
              " vs ", num_trailing_blank_frames->size());

  auto device = model.Device();
  int32_t blank_id = model.BlankId();
  int32_t unk_id = model.UnkId();
  int32_t context_size = model.ContextSize();

  int32_t N = encoder_out.size(0);
  int32_t T = encoder_out.size(1);

  auto decoder_input =
      torch::full({N, context_size}, blank_id,
                  torch::dtype(torch::kLong)
                      .memory_format(torch::MemoryFormat::Contiguous));

  encoder_out = model.ForwardEncoderProj(encoder_out);

  for (int32_t t = 0; t != T; ++t) {
    auto cur_encoder_out = encoder_out.index({torch::indexing::Slice(), t});

    auto logits = model.ForwardJoiner(cur_encoder_out, decoder_out);
    auto max_indices = logits.argmax(/*dim*/ -1).cpu();
    auto max_indices_accessor = max_indices.accessor<int64_t, 1>();
    bool emitted = false;
    for (int32_t n = 0; n != N; ++n) {
      auto index = max_indices_accessor[n];
      if (index != blank_id && index != unk_id) {
        emitted = true;
        (*hyps)[n].push_back(index);
        (*num_trailing_blank_frames)[n] = 0;
      } else {
        (*num_trailing_blank_frames)[n] += 1;
      }
    }

    if (emitted) {
      BuildDecoderInput(*hyps, &decoder_input);
      decoder_out = model.ForwardDecoder(decoder_input.to(device)).squeeze(1);
      decoder_out = model.ForwardDecoderProj(decoder_out);
    }
  }
  return decoder_out;
}

std::vector<std::vector<int32_t>> ModifiedBeamSearch(
    RnntModel &model,  // NOLINT
    torch::Tensor encoder_out, torch::Tensor encoder_out_length,
    int32_t num_active_paths /*=4*/) {
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

  torch::Device device = model.Device();
  encoder_out = encoder_out.to(device);

  torch::nn::utils::rnn::PackedSequence packed_seq =
      torch::nn::utils::rnn::pack_padded_sequence(encoder_out,
                                                  encoder_out_length,
                                                  /*batch_first*/ true,
                                                  /*enforce_sorted*/ false);

  auto projected_encoder_out = model.ForwardEncoderProj(packed_seq.data());

  int32_t blank_id = model.BlankId();
  int32_t unk_id = model.UnkId();
  int32_t context_size = model.ContextSize();

  int32_t batch_size = encoder_out_length.size(0);

  std::vector<int32_t> blanks(context_size, blank_id);
  Hypotheses blank_hyp({{blanks, 0}});

  std::deque<Hypotheses> finalized;
  std::vector<Hypotheses> cur(batch_size, blank_hyp);
  std::vector<Hypothesis> prev;

  using torch::indexing::Slice;
  auto batch_sizes_acc = packed_seq.batch_sizes().accessor<int64_t, 1>();
  int32_t num_batches = packed_seq.batch_sizes().numel();
  int32_t offset = 0;

  for (int32_t i = 0; i != num_batches; ++i) {
    int32_t cur_batch_size = batch_sizes_acc[i];
    int32_t start = offset;
    int32_t end = start + cur_batch_size;
    auto cur_encoder_out = projected_encoder_out.index({Slice(start, end)});
    offset = end;

    cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // Now cur_encoder_out's shape is (cur_batch_size, 1, 1, joiner_dim)

    if (cur_batch_size < cur.size()) {
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
    for (int32_t k = 0; k != prev.size(); ++k) {
      ys_log_probs_acc[k][0] = prev[k].log_prob;
    }

    auto decoder_input = BuildDecoderInput(prev, context_size).to(device);

    auto decoder_out = model.ForwardDecoder(decoder_input);
    decoder_out = model.ForwardDecoderProj(decoder_out);
    // decoder_out is of shape (num_hyps, 1, joiner_dim)

    auto index = k2::RowIds(hyps_shape, 1).to(torch::kLong).to(device);

    cur_encoder_out = cur_encoder_out.index_select(/*dim*/ 0, /*index*/ index);
    // cur_encoder_out is of shape (num_hyps, 1, 1, joiner_dim)

    auto logits =
        model.ForwardJoiner(cur_encoder_out, decoder_out.unsqueeze(1));

    // logits' shape is (num_hyps, 1, 1, vocab_size)
    logits = logits.squeeze(1).squeeze(1);
    // now logits' shape is (num_hyps, vocab_size)

    auto log_probs = logits.log_softmax(-1).cpu();

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
              .topk(/*k*/ num_active_paths, /*dim*/ 0,
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
        if (new_token != blank_id && new_token != unk_id) {
          new_hyp.ys.push_back(new_token);
        }

        // We already added log_prob of the path to log_probs before, so
        // we use values_acc[j] here directly.
        new_hyp.log_prob = values_acc[j];
        hyps.Add(std::move(new_hyp));
      }
      cur.push_back(std::move(hyps));
    }
  }

  for (auto &h : finalized) {
    cur.push_back(std::move(h));
  }

  auto unsorted_indices = packed_seq.unsorted_indices().cpu();
  auto unsorted_indices_accessor = unsorted_indices.accessor<int64_t, 1>();

  std::vector<std::vector<int32_t>> ans(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    Hypothesis hyp = cur[unsorted_indices_accessor[i]].GetMostProbable(true);
    torch::ArrayRef<int32_t> arr(hyp.ys);
    ans[i] = arr.slice(context_size).vec();
  }

  return ans;
}

std::vector<Hypotheses> StreamingModifiedBeamSearch(
    RnntModel &model,  // NOLINT
    torch::Tensor encoder_out, std::vector<Hypotheses> in_hyps,
    int32_t num_active_paths /*= 4*/) {
  TORCH_CHECK(encoder_out.dim() == 3, encoder_out.dim(), " vs ", 3);

  auto device = model.Device();
  int32_t blank_id = model.BlankId();
  int32_t unk_id = model.UnkId();
  int32_t context_size = model.ContextSize();

  int32_t N = encoder_out.size(0);
  int32_t T = encoder_out.size(1);

  encoder_out = model.ForwardEncoderProj(encoder_out);

  std::vector<Hypotheses> cur = std::move(in_hyps);
  std::vector<Hypothesis> prev;

  for (int32_t t = 0; t != T; ++t) {
    auto cur_encoder_out = encoder_out.index({torch::indexing::Slice(), t});
    cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // Now cur_encoder_out's shape is (N, 1, 1, joiner_dim)

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
    auto decoder_out = model.ForwardDecoder(decoder_input);
    decoder_out = model.ForwardDecoderProj(decoder_out);
    // decoder_out is of shape (num_hyps, 1, joiner_dim)

    auto index = k2::RowIds(hyps_shape, 1).to(torch::kLong).to(device);
    cur_encoder_out = cur_encoder_out.index_select(/*dim*/ 0, /*index*/ index);
    // cur_encoder_out is of shape (num_hyps, 1, 1, joiner_dim)

    auto logits =
        model.ForwardJoiner(cur_encoder_out, decoder_out.unsqueeze(1));
    // logits' shape is (num_hyps, 1, 1, vocab_size)
    logits = logits.squeeze(1).squeeze(1);
    // now logits' shape is (num_hyps, vocab_size)

    auto log_probs = logits.log_softmax(-1).cpu();

    log_probs.add_(ys_log_probs);

    int32_t vocab_size = log_probs.size(1);
    log_probs = log_probs.reshape(-1);
    auto row_splits = k2::RowSplits(hyps_shape, 1);
    auto row_splits_acc = row_splits.accessor<int32_t, 1>();

    for (int32_t k = 0; k != N; ++k) {
      int32_t start = row_splits_acc[k];
      int32_t end = row_splits_acc[k + 1];

      torch::Tensor values, indexes;
      std::tie(values, indexes) =
          log_probs.slice(/*dim*/ 0, start * vocab_size, end * vocab_size)
              .topk(/*k*/ num_active_paths, /*dim*/ 0,
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
        if (new_token != blank_id && new_token != unk_id) {
          new_hyp.ys.push_back(new_token);
          new_hyp.num_trailing_blanks = 0;
        } else {
          new_hyp.num_trailing_blanks += 1;
        }

        // We already added log_prob of the path to log_probs before, so
        // we use values_acc[j] here directly.
        new_hyp.log_prob = values_acc[j];
        hyps.Add(std::move(new_hyp));
      }
      cur.push_back(std::move(hyps));
    }  // for (int32_t k = 0; k != N; ++k)
  }    // for (int32_t t = 0; t != T; ++t)

  return cur;
}

}  // namespace sherpa
