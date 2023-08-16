// sherpa/csrc/online-transducer-fast-beam-search-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/online-transducer-fast-beam-search-decoder.h"

#include <utility>

#include "k2/torch_api.h"
#include "sherpa/csrc/online-transducer-decoder.h"

namespace sherpa {

OnlineTransducerFastBeamSearchDecoder::OnlineTransducerFastBeamSearchDecoder(
    OnlineTransducerModel *model, const FastBeamSearchConfig &config)
    : model_(model), config_(config), vocab_size_(model->VocabSize()) {
  if (config.lg.empty()) {
    // Use a trivial graph
    decoding_graph_ = k2::GetTrivialGraph(vocab_size_ - 1, model_->Device());
  } else {
    decoding_graph_ = k2::LoadFsaClass(config.lg, model_->Device());
    k2::ScaleTensorAttribute(decoding_graph_, config.ngram_lm_scale, "scores");
  }
}

OnlineTransducerDecoderResult
OnlineTransducerFastBeamSearchDecoder::GetEmptyResult() {
  OnlineTransducerDecoderResult r;
  r.rnnt_stream = k2::CreateRnntStream(decoding_graph_);
  return r;
}

void OnlineTransducerFastBeamSearchDecoder::Decode(
    torch::Tensor encoder_out,
    std::vector<OnlineTransducerDecoderResult> *results) {
  TORCH_CHECK(encoder_out.dim() == 3, encoder_out.dim(), " vs ", 3);

  TORCH_CHECK(encoder_out.size(0) == static_cast<int32_t>(results->size()),
              encoder_out.size(0), " vs ", results->size());

  auto device = model_->Device();
  int32_t context_size = model_->ContextSize();

  std::vector<k2::RnntStreamPtr> stream_vec;
  std::vector<int32_t> num_processed_frames_vec;

  stream_vec.reserve(results->size());
  num_processed_frames_vec.reserve(results->size());

  for (auto &r : *results) {
    stream_vec.push_back(r.rnnt_stream);

    // number of frames before subsampling
    num_processed_frames_vec.push_back(r.num_processed_frames);
  }

  torch::Tensor num_processed_frames =
      torch::tensor(num_processed_frames_vec, torch::kInt);

  k2::RnntStreamsPtr streams =
      k2::CreateRnntStreams(stream_vec, vocab_size_, context_size, config_.beam,
                            config_.max_contexts, config_.max_states);

  int32_t N = encoder_out.size(0);
  int32_t T = encoder_out.size(1);
  k2::RaggedShapePtr shape;
  torch::Tensor contexts;
  for (int32_t t = 0; t != T; ++t) {
    std::tie(shape, contexts) = k2::GetRnntContexts(streams);
    contexts = contexts.to(torch::kLong);
    // contexts.shape: (num_hyps, context_size)

    auto decoder_out = model_->RunDecoder(contexts).squeeze(1);
    // decoder_out.shape: (num_hyps, joiner_dim)

    auto cur_encoder_out = encoder_out.index({torch::indexing::Slice(), t});
    // cur_encoder_out has shape (N, joiner_dim)

    auto index = k2::RowIds(shape, 1).to(torch::kLong).to(device);
    cur_encoder_out = cur_encoder_out.index_select(/*dim*/ 0, /*index*/ index);
    // cur_encoder_out has shape (num_hyps, joiner_dim)

    auto logits = model_->RunJoiner(cur_encoder_out, decoder_out);
    // logits.shape: (num_hyps, vocab_size)

    auto log_probs = logits.log_softmax(-1);
    k2::AdvanceRnntStreams(streams, log_probs);
  }  // for (int32_t t = 0; t != T; ++t)

  k2::TerminateAndFlushRnntStreams(streams);

  // TODO(fangjun): This assumes the subsampling factor is 4
  num_processed_frames = (num_processed_frames / 4).to(torch::kInt) + T;

  std::vector<int32_t> processed_frames_vec(
      num_processed_frames.data_ptr<int32_t>(),
      num_processed_frames.data_ptr<int32_t>() + num_processed_frames.numel());

  auto lattice =
      k2::FormatOutput(streams, processed_frames_vec, config_.allow_partial);

  lattice = k2::ShortestPath(lattice);

  // Get tokens and timestamps from the lattice
  auto labels = k2::GetTensorAttr(lattice, "labels").cpu().contiguous();
  auto acc = labels.accessor<int32_t, 1>();

  for (auto &r : *results) {
    r.tokens.clear();
    r.timestamps.clear();
    r.num_trailing_blanks = 0;
  }
  OnlineTransducerDecoderResult *p = results->data();

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
      ++p->num_trailing_blanks;
      continue;
    }

    p->num_trailing_blanks = 0;
    p->tokens.push_back(token);
    p->timestamps.push_back(t);
    ++t;
  }  // for (int32_t i = 0, t = 0; i != labels.numel(); ++i)
}

}  // namespace sherpa
