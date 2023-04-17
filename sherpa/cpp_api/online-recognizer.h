// sherpa/cpp_api/online-recognizer.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_ONLINE_RECOGNIZER_H_
#define SHERPA_CPP_API_ONLINE_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa/cpp_api/endpoint.h"
#include "sherpa/cpp_api/fast-beam-search-config.h"
#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/online-stream.h"

namespace sherpa {

struct OnlineRecognizerConfig {
  /// Config for the feature extractor
  FeatureConfig feat_config;

  EndpointConfig endpoint_config;

  FastBeamSearchConfig fast_beam_search_config;

  /// Path to the torchscript model
  std::string nn_model;

  /// Path to tokens.txt
  std::string tokens;

  // The following three are for RnntLstmModel
  std::string encoder_model;
  std::string decoder_model;
  std::string joiner_model;

  /// true to use GPU for neural network computation and decoding.
  /// false to use CPU.
  /// You can use CUDA_VISIBLE_DEVICES to control which device to use.
  /// We always use GPU 0 in the code. This also implies it supports only
  /// 1 GPU at present.
  /// Note: You have to use a CUDA version of PyTorch in order to use
  /// GPU for computation
  bool use_gpu = false;

  bool use_endpoint = false;

  std::string decoding_method = "greedy_search";

  /// used only for modified_beam_search
  int32_t num_active_paths = 4;

  // For OnlineConformerTransducerModel, i.e., for models from
  // pruned_transducer_stateless{2,3,4,5} in icefall
  // In number of frames after subsampling
  int32_t left_context = 64;

  // For OnlineConformerTransducerModel, i.e., for models from
  // pruned_transducer_stateless{2,3,4,5} in icefall
  // In number of frames after subsampling
  int32_t right_context = 0;

  // For OnlineConformerTransducerModel, i.e., for models from
  // pruned_transducer_stateless{2,3,4,5} in icefall
  // In number of frames after subsampling
  int32_t chunk_size = 12;

  void Register(ParseOptions *po);

  void Validate() const;

  /** A string representation for debugging purpose. */
  std::string ToString() const;
};

class OnlineRecognizer {
 public:
  /** Construct an instance of OnlineRecognizer.
   *
   * @param config Configuration for the recognizer.
   */
  explicit OnlineRecognizer(const OnlineRecognizerConfig &config);

  ~OnlineRecognizer();

  const OnlineRecognizerConfig &GetConfig() const;

  // Create a stream for decoding.
  std::unique_ptr<OnlineStream> CreateStream();

  /**
   * Return true if the given stream has enough frames for decoding.
   * Return false otherwise
   */
  bool IsReady(OnlineStream *s);

  /**
   * Return true if VAD activity
   * Return false otherwise
   */
  bool IsEndpoint(OnlineStream *s);

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

  OnlineRecognitionResult GetResult(OnlineStream *s);

 private:
  class OnlineRecognizerImpl;
  std::unique_ptr<OnlineRecognizerImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_ONLINE_RECOGNIZER_H_
