// sherpa/cpp_api/offline-stream.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CPP_API_OFFLINE_STREAM_H_
#define SHERPA_CPP_API_OFFLINE_STREAM_H_

#include <memory>
#include <string>
#include <vector>

#include "kaldifeat/csrc/feature-fbank.h"
#include "torch/script.h"

namespace sherpa {

struct OfflineRecognitionResult {
  // Recognition results.
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

class OfflineStream {
 public:
  ~OfflineStream();

  /** Create a stream.
   *
   * @param fbank Not owned by this class.
   */
  explicit OfflineStream(kaldifeat::Fbank *fbank);

  /** Create a stream from a WAVE file.
   *
   * @param wave_file Path to the WAVE file. Its sample frequency should
   *                  match the one from the feature extractor. Only
   *                  WAVEs with a single channel are supported.
   */
  void AcceptWaveFile(const std::string &wave_file);

  /** Create a stream from audio samples.
   *
   * @param fbank_
   * @param samples Pointer to the audio samples. Whether it should be
   *                normalized depends on how the model is trained.
   *                For models from icefall, it should be normalized
   *                to [-1, 1] before passing to this function.
   * @param n  Number of audio samples.
   */
  void AcceptSamples(const float *samples, int32_t n);

  /** Create a stream from features.
   *
   * @param feature Pointer to the 2-D feature matrix of shape
   *                [num_frames][num_channels]. It should be contiguous
   *                in memory and stored in row major.
   * @param num_frames Number of feature frames.
   * @param num_channels It should match the one from the feature extractor.
   */
  void AcceptFeatures(const float *feature, int32_t num_frames,
                      int32_t num_channels);

  /** Get the features of this stream.
   *
   * @return Return a 2-D tensor of shape (num_frames, num_channels).
   */
  const torch::Tensor &GetFeatures() const;

  /** Set the recognition result for this stream. */
  void SetResult(const OfflineRecognitionResult &r);

  /** Get the recognition result of this stream */
  const OfflineRecognitionResult &GetResult() const;

 private:
  class OfflineStreamImpl;
  std::unique_ptr<OfflineStreamImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_STREAM_H_
