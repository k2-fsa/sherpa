/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

/**
Online ASR APIs for sherpa.

Note: It supports only models from
https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2
at present.

You can find a pre-trained model in the following address:

https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
 */
#ifndef SHERPA_CPP_API_ONLINE_RECOGNIZER_H_
#define SHERPA_CPP_API_ONLINE_RECOGNIZER_H_

#include <memory>
#include <string>

namespace sherpa {

class OnlineStream;

enum class DecodingMethod {
  kGreedySearch = 0,
  kModifiedBeamSearch = 1,
};

constexpr auto kGreedySearch = DecodingMethod::kGreedySearch;
constexpr auto kModifiedBeamSearch = DecodingMethod::kModifiedBeamSearch;

struct DecodingOptions {
  DecodingMethod method = kGreedySearch;
  // kGreedySearch has no options

  // Options for kModifiedBeamSearch
  int32_t num_active_paths = 4;

  // For RnntConformerModel, i.e., for models from
  // pruned_transducer_statelessX in icefall
  // In number of frames after subsampling
  int32_t left_context = -1;

  // For RnntConformerModel, i.e., for models from
  // pruned_transducer_statelessX in icefall
  // In number of frames after subsampling
  int32_t right_context = -1;

  // For RnntConformerModel, i.e., for models from
  // pruned_transducer_statelessX in icefall
  // In number of frames after subsampling
  int32_t chunk_size = -1;
};

struct OnlineRecognitionResult {
  // Recognition results.
  // For English, it consists of space separated words.
  // For Chinese, it consists of Chinese words without spaces.
  std::string text;
  std::string AsJsonString() const;
};

class OnlineRecognizer {
 public:
  /** Construct an instance of OnlineRecognizer.
   *
   * @param nn_model  Path to the torchscript model. We assume the model
   *                  is one of conv_emformer_transducer_stateless2 from
   *                  icefall.
   *
   * @param tokens    Path to the tokens.txt. Each line in this file has
   *                  two columns separated by space(s). The first column is
   *                  a symbol while the second column is the integer ID of
   *                  the symbol. If you have a bpe.model, please convert it
   *                  to tokens.txt first.
   * @param decoding_opts   Decoding options for this recognizer.
   * @param use_gpu         true to use GPU for neural network computation.
   *                        false to use CPU. If true, we always select GPU 0.
   *                        You can use the environment variable
   *                        CUDA_VISIBLE_DEVICES to control which device should
   *                        be mapped to GPU 0.
   * @param sample_rate     The expected audio sample rate of the model.
   */
  OnlineRecognizer(const std::string &nn_model, const std::string &tokens,
                   const DecodingOptions &decoding_opts = {},
                   bool use_gpu = false, float sample_rate = 16000);

  /** Construct an instance of OnlineRecognizer.
   *
   * @param encoder_model  Path to the encoder model. We assume the model
   *                       is one of lstm_transducer_statelessX from icefall.
   *
   * @param decoder_model  Path to the decoder model. We assume the model
   *                       is one of lstm_transducer_statelessX from icefall.
   *
   * @param joiner_model  Path to the joiner model. We assume the model
   *                       is one of lstm_transducer_statelessX from icefall.
   *
   * @param tokens    Path to the tokens.txt. Each line in this file has
   *                  two columns separated by space(s). The first column is
   *                  a symbol while the second column is the integer ID of
   *                  the symbol. If you have a bpe.model, please convert it
   *                  to tokens.txt first.
   * @param decoding_opts   Decoding options for this recognizer.
   * @param use_gpu         true to use GPU for neural network computation.
   *                        false to use CPU. If true, we always select GPU 0.
   *                        You can use the environment variable
   *                        CUDA_VISIBLE_DEVICES to control which device should
   *                        be mapped to GPU 0.
   * @param sample_rate     The expected audio sample rate of the model.
   */
  OnlineRecognizer(const std::string &encoder_model,
                   const std::string &decoder_model,
                   const std::string &joiner_model, const std::string &tokens,
                   const DecodingOptions &decoding_opts = {},
                   bool use_gpu = false, float sample_rate = 16000);

  ~OnlineRecognizer();

  // Create a stream for decoding.
  std::unique_ptr<OnlineStream> CreateStream();

  /**
   * Return true if the given stream has enough frames for decoding.
   * Return false otherwise
   */
  bool IsReady(OnlineStream *s);

  /** Decode a single stream. */
  void DecodeStream(OnlineStream *s) {
    OnlineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode multiple streams in parallel
   *
   * @param ss Pointer array containing streams to be decoded.
   * @param n Number of streams in `ss`.
   */
  void DecodeStreams(OnlineStream **ss, int32_t n);

  OnlineRecognitionResult GetResult(OnlineStream *s) const;

 private:
  class OnlineRecognizerImpl;
  std::unique_ptr<OnlineRecognizerImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_ONLINE_RECOGNIZER_H_
