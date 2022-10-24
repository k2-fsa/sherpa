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
#ifndef SHERPA_CSRC_ONLINE_ASR_H_
#define SHERPA_CSRC_ONLINE_ASR_H_

#include <memory>
#include <string>

#include "kaldifeat/csrc/feature-fbank.h"
#include "sherpa/cpp_api/online_stream.h"
#include "sherpa/csrc/endpoint.h"
#include "sherpa/csrc/parse_options.h"
#include "sherpa/csrc/rnnt_model.h"
#include "sherpa/csrc/symbol_table.h"
#include "torch/script.h"

namespace sherpa {

struct OnlineAsrOptions {
  /// Path to torchscript model.
  /// It is for the following models:
  ///  - RnntConvEmformerModel
  std::string nn_model;

  // The following three are for RnntLstmModel
  std::string encoder_model;
  std::string decoder_model;
  std::string joiner_model;

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

  EndpointConfig endpoint_config;

  void Register(ParseOptions *po);

  // Check that option values are valid
  void Validate() const;

  // For debugging
  std::string ToString() const;
};

class OnlineAsr {
 public:
  explicit OnlineAsr(const OnlineAsrOptions &opts);
  ~OnlineAsr() = default;
  const OnlineAsrOptions &Opts() const { return opts_; }

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

  // TODO(fangjun): Return a struct
  std::string GetResult(OnlineStream *s) const;

 private:
  void GreedySearch(OnlineStream **ss, int32_t n);

  void ModifiedBeamSearch(OnlineStream **ss, int32_t n);

  std::string GetGreedySearchResult(OnlineStream *s) const;
  std::string GetModifiedBeamSearchResult(OnlineStream *s) const;

 private:
  OnlineAsrOptions opts_;
  // TODO(fangjun): Change it to std::unique_ptr<RnntModel>
  std::unique_ptr<RnntModel> model_;

  SymbolTable sym_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_ONLINE_ASR_H_
