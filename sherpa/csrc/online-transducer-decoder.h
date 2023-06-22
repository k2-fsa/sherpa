// sherpa/csrc/online-transducer-decoder.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CSRC_ONLINE_TRANSDUCER_DECODER_H_
#define SHERPA_CSRC_ONLINE_TRANSDUCER_DECODER_H_

#include <vector>

#include "k2/torch_api.h"
#include "sherpa/cpp_api/online-stream.h"
#include "sherpa/csrc/hypothesis.h"
#include "torch/script.h"

namespace sherpa {

struct OnlineTransducerDecoderResult {
  /// Number of frames we have decoded so far
  int32_t frame_offset = 0;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  /// The decoded token IDs so far
  std::vector<int32_t> tokens;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;

  // used only for modified_beam_search
  Hypotheses hyps;

  // used only for fast_beam_search
  k2::RnntStreamPtr rnnt_stream;

  // Before subsampling. Used only for fast_beam_search
  int32_t num_processed_frames = 0;
};

class OnlineTransducerDecoder {
 public:
  virtual ~OnlineTransducerDecoder() = default;

  /* Return an empty result.
   *
   * To simplify the decoding code, we add `context_size` blanks
   * to the beginning of the decoding result, which will be
   * stripped by calling `StripPrecedingBlanks()`.
   */
  virtual OnlineTransducerDecoderResult GetEmptyResult() = 0;

  /** Strip blanks added by `GetEmptyResult()`. */
  virtual void StripLeadingBlanks(OnlineTransducerDecoderResult * /*r*/) {}

  /* Finalize the context graph searching, it will subtract the bonus of
   * partial matching hypothesis.
   *
   * Used only in modified_beam_search and when context_graph is given.
   */
  virtual void FinalizeResult(OnlineStream * /*s*/,
                              OnlineTransducerDecoderResult * /*r*/) {}

  /** Run transducer beam search given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   *
   * @note This is no need to pass encoder_out_length here since for the
   * online decoding case, each utterance has the same number of frames
   * and there are no paddings.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual void Decode(torch::Tensor encoder_out,
                      std::vector<OnlineTransducerDecoderResult> *result) = 0;

  virtual void Decode(torch::Tensor encoder_out, OnlineStream **ss,
                      int32_t num_streams,
                      std::vector<OnlineTransducerDecoderResult> *result) {
    SHERPA_LOG(FATAL) << "This interface is for ModifiedBeamSearchDecoder.";
  }
};
}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_TRANSDUCER_DECODER_H_
