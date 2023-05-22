// sherpa/csrc/offline-transducer-fast-beam-search-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/csrc/offline-transducer-fast-beam-search-decoder.h"

#include <utility>

#include "k2/torch_api.h"

namespace sherpa {

OfflineTransducerFastBeamSearchDecoder::OfflineTransducerFastBeamSearchDecoder(
    OfflineTransducerModel *model, const FastBeamSearchConfig &config)
    : model_(model), config_(config), vocab_size_(model->VocabSize()) {
  if (config.lg.empty()) {
    // Use a trivial graph
    decoding_graph_ = k2::GetTrivialGraph(vocab_size_ - 1, model_->Device());
  } else {
    decoding_graph_ = k2::LoadFsaClass(config.lg, model_->Device());
    k2::ScaleTensorAttribute(decoding_graph_, config.ngram_lm_scale, "scores");
  }
}

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerFastBeamSearchDecoder::Decode(
    torch::Tensor encoder_out, torch::Tensor encoder_out_length) {
  TORCH_CHECK(encoder_out.dim() == 3, encoder_out.dim(), " vs ", 3);

  auto device = model_->Device();
  int32_t context_size = model_->ContextSize();

  int32_t batch_size = encoder_out.size(0);
  int32_t num_frames = encoder_out.size(1);

  std::vector<k2::RnntStreamPtr> stream_vec;
  stream_vec.reserve(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    stream_vec.push_back(k2::CreateRnntStream(decoding_graph_));
  }

  k2::RnntStreamsPtr streams =
      k2::CreateRnntStreams(stream_vec, vocab_size_, context_size, config_.beam,
                            config_.max_contexts, config_.max_states);

  k2::RaggedShapePtr shape;
  torch::Tensor contexts;

  for (int32_t t = 0; t != num_frames; ++t) {
    std::tie(shape, contexts) = k2::GetRnntContexts(streams);
    contexts = contexts.to(torch::kLong);
    // contexts.shape: (num_hyps, context_size)

    auto decoder_out = model_->RunDecoder(contexts).unsqueeze(1);
    // decoder_out.shape: (num_hyps, 1, 1, joiner_dim)

    auto cur_encoder_out = encoder_out.index({torch::indexing::Slice(), t});
    // cur_encoder_out has shape (N, joiner_dim)

    auto index = k2::RowIds(shape, 1).to(torch::kLong).to(device);
    cur_encoder_out = cur_encoder_out.index_select(/*dim*/ 0, /*index*/ index);
    // cur_encoder_out has shape (num_hyps, joiner_dim)

    cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // cur_encoder_out.shape (num_hyps, 1, 1, joiner_dim)

    auto logits = model_->RunJoiner(cur_encoder_out, decoder_out);
    // logits.shape: (num_hyps, 1, 1, vocab_size)

    logits = logits.squeeze(1).squeeze(1);
    // logits.shape: (num_hyps, vocab_size)

    auto log_probs = logits.log_softmax(-1);
    k2::AdvanceRnntStreams(streams, log_probs);
  }

  k2::TerminateAndFlushRnntStreams(streams);

  encoder_out_length = encoder_out_length.cpu().to(torch::kInt);
  std::vector<int32_t> processed_frames_vec(
      encoder_out_length.data_ptr<int32_t>(),
      encoder_out_length.data_ptr<int32_t>() + encoder_out_length.numel());

  auto lattice =
      k2::FormatOutput(streams, processed_frames_vec, config_.allow_partial);

  lattice = k2::ShortestPath(lattice);

  std::vector<OfflineTransducerDecoderResult> results(batch_size);

  // Get tokens and timestamps from the lattice
  auto labels = k2::GetTensorAttr(lattice, "labels").cpu().contiguous();
  auto acc = labels.accessor<int32_t, 1>();

  OfflineTransducerDecoderResult *p = results.data();

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

    p->tokens.push_back(token);
    p->timestamps.push_back(t);
    ++t;
  }  // for (int32_t i = 0, t = 0; i != labels.numel(); ++i)

  return results;
}

}  // namespace sherpa
