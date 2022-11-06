// sherpa/cpp_api/offline-recognizer.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-stream.h"

namespace sherpa {

struct OfflineRecognizerConfig {
  /// Config for the feature extractor
  FeatureConfig feat_config;

  /// Path to the torchscript model
  std::string nn_model;

  /// Path to tokens.txt
  std::string tokens;

  /// true to use GPU for neural network computation and decoding.
  /// false to use CPU.
  /// You can use CUDA_VISIBLE_DEVICES to control which device to use.
  /// We always use GPU 0 in the code. This also implies it supports only
  /// 1 GPU at present.
  /// Note: You have to use a CUDA version of PyTorch in order to use
  /// GPU for computation
  bool use_gpu = false;

  std::string decoding_method = "greedy_search";

  /// used only for modified_beam_search
  int32_t num_active_paths = 4;

  void Register(ParseOptions *po);

  void Validate() const;

  /** A string representation for debugging purpose. */
  std::string ToString() const;
};

std::ostream &operator<<(std::ostream &os,
                         const OfflineRecognizerConfig &config);

class OfflineRecognizer {
 public:
  ~OfflineRecognizer();

  explicit OfflineRecognizer(const OfflineRecognizerConfig &config);

  /// Create a stream for decoding.
  std::unique_ptr<OfflineStream> CreateStream();

  /** Decode a single stream
   *
   * @param s The stream to decode.
   */
  void DecodeStream(OfflineStream *s) {
    OfflineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode a list of streams.
   *
   * @param ss Pointer to an array of streams.
   * @param n  Size of the input array.
   */
  void DecodeStreams(OfflineStream **ss, int32_t n);

  /** Get the recognition result of the given stream.
   *
   * @param s The stream to get the result.
   * @return Return the recognition result for `s`.
   */
  OfflineRecognitionResult GetResult(OfflineStream *s) const;

 private:
  class OfflineRecognizerImpl;
  std::unique_ptr<OfflineRecognizerImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_
