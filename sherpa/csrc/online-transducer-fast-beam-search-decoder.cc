// sherpa/csrc/online-transducer-fast-beam-search-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "k2/torch_api.h"
#include "sherpa/csrc/online-transducer-decoder.h"

namespace sherap {

OnlineTransducerFastBeamSearchDecoder::OnlineTransducerFastBeamSearchDecoder(
    OnlineTransducerModel model, const FastBeamSearchConfig &config,
    int32_t vocab_size, torch::Device device)
    : model_(model), config_(config), vocab_size_(vocab_size) {
  if (config.lg.empty()) {
    // Use a trivial graph
    decoding_graph_ = k2::GetTrivialGraph(vocab_size - 1, device);
  } else {
    decoding_graph_ = k2::LoadFsaClass(config.lg, device);
    // TODO(fangjun): Scale decoding_graph_.scores with config.ngram_lm_scale
  }
}

OnlineTransducerDecoderResult
OnlineTransducerFastBeamSearchDecoder::GetEmptyResult() {
  OnlineTransducerDecoderResult r;
  r.rnnt_stream = k2::CreateRnntStream(decoding_graph_);
  return r;
}

void OnlineTransducerFastBeamSearchDecoder::StripLeadingBlanks(
    OnlineTransducerDecoderResult *r) {
  // TODO(fangjun): Implement it
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

  for (auto *r : results) {
    stream_vec.push_back(r->rnnt_stream);

    // number of frames before subsampling
    num_processed_frames_vec.push_back(r->num_processed_frames);
  }

  torch::Tensor num_processed_frames =
      torch::tensor(num_processed_frames_vec, torch::kInt).to(device);

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
  }
  k2::TerminateAndFlushRnntStreams(streams);

  // TODO(fangjun): This assumes the subsampling factor is 4
  num_processed_frames = (num_processed_frames / 4).to(torch::kInt) + T;

  auto lattice =
      k2::FormatOutput(streams, num_processed_frames, config_.allow_partial);
  std::vector<std::vector<int32_t>> tokens = k2::BestPath(lattice);
  for (int32_t i = 0; i != N; ++i) {
    (*results)[i].tokens = std::move(tokens[i]);
  }
}

}  // namespace sherap
