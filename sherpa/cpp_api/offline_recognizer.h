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
#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

namespace sherpa {

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
};

struct OfflineRecognitionResult {
  // RecognitionResult results.
  // For English, it consists of space separated words.
  // For Chinese, it consists of Chinese words without spaces.
  std::string text;

  // Decoded results at the token level.
  // For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<int32_t> tokens;

  // timestamps.size() == tokens.size()
  // timestamps[i] records the frame number on which tokens[i] is decoded.
  // Frame numbers are counted after model subsampling.
  std::vector<int32_t> timestamps;  // not implemented at present
};

class OfflineRecognizer {
 public:
  /** Construct an instance of OfflineRecognizer.
   *
   * @param nn_model  Path to the torchscript model. We assume the model
   *                  is one of pruned_transducer_statelessX from icefall.
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
  OfflineRecognizer(const std::string &nn_model, const std::string &tokens,
                    const DecodingOptions &decoding_opts = {},
                    bool use_gpu = false, float sample_rate = 16000);

  ~OfflineRecognizer();

  /** Decode a single file.
   *
   * Only ".wav" format is supported. If the input wave file has multiple
   * channels, only the first channel is used.
   *
   * Note that the sample rate of the input wave file must match the one
   * expected by the model. No resampling is done if they differ. Instead
   * it will abort.
   *
   * @param filename Path to the wave file.
   *
   * @return Return the recognition result.
   */
  OfflineRecognitionResult DecodeFile(const std::string &filename) {
    return DecodeFileBatch({filename})[0];
  }

  /** Decode a batch of files.
   *
   * Only ".wav" format is supported. If the input wave file has multiple
   * channels, only the first channel is used.
   *
   * Note that the sample rate of the input wave file must match the one
   * expected by the model. No resampling is done if they differ. Instead
   * it will abort.
   *
   * @param filenames A list of paths to the waves files to be decoded.
   *
   * @return Return a list of recognition results. ans[i] is the results for
   *         filenames[i].
   */
  std::vector<OfflineRecognitionResult> DecodeFileBatch(
      const std::vector<std::string> &filenames);

  /** Decode audio samples.
   *
   * The sample rate of the input samples should match the one expected
   * by the model, which is 16 kHz for models from icefall.
   *
   * @param samples Pointer to a 1-D array of length `N` containing audio
   *                samples which should be normalized to the range [-1, 1]
   *                if you use a model from icefall. It should be on CPU.
   *
   * @param n  Length of the input samples.
   *
   * @return Return the recognition result.
   */
  OfflineRecognitionResult DecodeSamples(const float *samples, int32_t n) {
    const float *samples_array[1] = {samples};
    return DecodeSamplesBatch(samples_array, &n, 1)[0];
  }

  /** Decode a batch of audio samples
   *
   * The sample rate of the input samples should match the one expected
   * by the model, which is 16 kHz for models from icefall.
   *
   * @param samples Pointer to a 1-D array of length `n` containing pointers to
   *                1-D arrays of audio samples. All samples should be on CPU.
   *
   * @param samples_length  Pointer to a 1-D array of length `n`.
   *                        samples_length[i] contains the number of samples
   *                        in samples[i]. It should be on CPU.
   *
   * @return Return the recognition results.
   */
  std::vector<OfflineRecognitionResult> DecodeSamplesBatch(
      const float **samples, const int32_t *samples_length, int32_t n);

  /** Decode fbank features.
   *
   * @param features Pointer to a 2-D array of shape (T, C). It is in row-major
   *                 and should be on CPU.
   * @param T Number of feature frames in `features`.
   * @param C Feature dimension which should match the one expected by the
   *          model.
   *
   * @return Return the recognition result.
   */
  OfflineRecognitionResult DecodeFeatures(const float *features, int32_t T,
                                          int32_t C) {
    return DecodeFeaturesBatch(features, &T, 1, T, C)[0];
  }

  /** Decode a batch of fbank features.
   *
   * @param features Pointer to a 3-D tensor of shape (N, T, C). It is in
   *                 row-major and should be on CPU.
   * @param features_length  Pointer to a 1-D tensor of shape (N,) containing
   *                         number of valid frames in `features` before
   *                         padding. It should be on CPU.
   * @param N Batch size.
   * @param T Number of feature frames.
   * @param C Feature dimension. It must match the one expected by the model.
   *
   * @return Return the recognition results.
   */
  std::vector<OfflineRecognitionResult> DecodeFeaturesBatch(
      const float *features, const int32_t *features_length, int32_t N,
      int32_t T, int32_t C);

 private:
  class OfflineRecognizerImpl;
  std::unique_ptr<OfflineRecognizerImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_
