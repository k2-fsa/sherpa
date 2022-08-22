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
#ifndef SHERPA_CSRC_OFFLINE_ASR_H_
#define SHERPA_CSRC_OFFLINE_ASR_H_

#include <string>
#include <vector>

#include "kaldifeat/csrc/feature-fbank.h"
#include "sherpa/csrc/parse_options.h"
#include "sherpa/csrc/rnnt_conformer_model.h"
#include "sherpa/csrc/symbol_table.h"

namespace sherpa {

struct OfflineAsrOptions {
  /// Path to torchscript model
  std::string nn_model;

  /// Path to tokens.txt.
  /// Each line the tokens.txt consists of two columms separated by a space:
  ///  - column 1: symbol
  ///  - column 2: integer ID of the symbol
  std::string tokens;

  /// Decoding method to use.
  /// Possible values are: greedy_search, modified_beam_search.
  std::string decoding_method = "greedy_search";

  /// Number of active paths in modified_beam_search.
  /// Used only when decoding_method is modified_beam_search.
  int32_t num_active_paths = 4;

  // true to use GPU for computation. Always selects the first device.
  // false to use CPU.
  // Note: Only neural network computation and decoding are done on CPU.
  // Feature extraction is performed on CPU.
  bool use_gpu = false;

  kaldifeat::FbankOptions fbank_opts;

  void Register(ParseOptions *po);

  // Check that option values are valid
  void Validate() const;

  // For debugging
  std::string ToString() const;
};

struct OfflineAsrResult {
  // Decoded results.
  // For English, it consists of space separated words.
  // For Chinese, it consists of Chinese words without spaces.
  std::string text;

  // Decoded results at the token level.
  // For BPE-based models, it consists of a list of BPE tokens.
  std::vector<int32_t> tokens;

  // timestamps.size() == tokens.size()
  // timestamps[i] records the frame number on which tokens[i] is decoded.
  // Frame numbers are counted after model subsampling.
  std::vector<int32_t> timestamps;
};

class OfflineAsr {
 public:
  explicit OfflineAsr(const OfflineAsrOptions &opts);

  /** Decode a single wave file.
   *
   * If the input wave has multiple channels, only the first channel is used
   * for decoding.
   *
   * @param filename Path to the wave file. Note: We only support "*.wav"
   *                 format.
   * @param expected_sample_rate  Expected sample rate of the input wave file.
   *                              If the input wave file has a different sample
   *                              rate from this value, it will abort.
   *
   * @return Return the recognition result.
   */
  OfflineAsrResult DecodeWave(const std::string &filename,
                              float expected_sample_rate) {
    return DecodeWaves({filename}, expected_sample_rate)[0];
  }

  /** Decode a batch of wave files in parallel.
   *
   * If an input wave has multiple channels, only the first channel is used
   * for decoding.
   *
   * @param filenames A list of wave filenames. We only support "*.wav" at
   *                  present.
   * @param expected_sample_rate  Expected sample rate of each input wave file.
   *                              If an input wave file has a different sample
   *                              rate from this value, it will abort.
   *
   * @return Return the recognition results. ans[i] contains the recognition
   *         result for filenames[i]
   */
  std::vector<OfflineAsrResult> DecodeWaves(
      const std::vector<std::string> &filenames, float expected_sample_rate);

  /** Decode audio samples.
   *
   * @param wave A 1-D torch.float32 tensor containing audio samples, which are
   *             normalized to the range [-1, 1). Its sample rate must match
   *             the one for the training data that is used to train the model.
   *             It is 16 kHz for all models trained by icefall.
   *
   * @return Return the recognition result.
   */
  OfflineAsrResult DecodeWave(torch::Tensor wave) {
    return DecodeWaves({wave})[0];
  }

  /** Decode a batch of audio samples in parallel.
   *
   * @param waves Each entry is a 1-D torch.float32 tensor containing audio
   *              samples, which are normalized to the range [-1, 1).
   *              Its sample rate must match the one for the training data that
   *              is used to train the model. It is 16 kHz for all models
   *              trained by icefall.
   *
   * @return Return the recognition result. ans[i] is the recognition result
   *         for wave[i].
   */
  std::vector<OfflineAsrResult> DecodeWaves(
      const std::vector<torch::Tensor> &waves);

  /** Decode input fbank feature.
   *
   * @param feature  A 2-D tensor containing the fbank feature of a wave. Its
   *                 number of rows equals to the number of feature frames and
   *                 the number of columns equals to the feature dimension.
   *
   * @return Return the recognition result.
   */
  OfflineAsrResult DecodeFeature(torch::Tensor feature) {
    return DecodeFeatures({feature})[0];
  }

  /** Decode a batch of input fbank features in parallel.
   *
   * @param features Each entry is a 2-D tensor containing the fbank feature
   *                 of a wave. Its number of rows equals to the number of
   *                 feature frames and the number of columns equals to the
   *                 feature dimension.
   *
   * @return Return the recognition result. ans[i] contains the recognition
   *         result for features[i].
   */
  std::vector<OfflineAsrResult> DecodeFeatures(
      const std::vector<torch::Tensor> &features);

  /** Decode from pre-computed features.
   *
   * @param features A 3-D tensor of shape (N, T, C) containing pre-computed
   *                 features.
   * @param features_length A 1-D tensor of shape (N,) containing number of
   *                        valid feature frames in `features` before padding.
   *
   * @return Return the recognition result. ans[i] contains the recognition
   *         result for features[i].
   */
  std::vector<OfflineAsrResult> DecodeFeatures(torch::Tensor features,
                                               torch::Tensor features_length);

 private:
  OfflineAsrOptions opts_;
  RnntConformerModel model_;
  SymbolTable sym_;

  kaldifeat::Fbank fbank_;  // always on CPU
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_ASR_H_
