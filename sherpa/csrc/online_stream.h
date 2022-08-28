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
#ifndef SHERPA_CSRC_ONLINE_STREAM_H_
#define SHERPA_CSRC_ONLINE_STREAM_H_

#include <cstdint>
#include <memory>

#include "torch/script.h"

namespace sherpa {

class OnlineStream {
 public:
  /**
   * @param sampling_rate The sampling rate for the feature extractor.
   *                      It should match the one used to train the model.
   * @param feature_dim  The feature dimension. It should match the one
   *                     used to train the model.
   * @param max_feature_vectors Number of feature frames to keep in the
   *                            recycling vector.
   *                            If it is set to -1, we keep all feature frames
   *                            computed so far.
   */
  OnlineStream(float sampling_rate, int32_t feature_dim,
               int32_t max_feature_vectors = -1);
  ~OnlineStream();

  /** This would be called from the application, when you get
   * more wave data.
   *
   * @param sampling_rate Sampling rate of the input waveform. It is provided so
   *                      the code can **assert** that it matches the sampling
   *                      rate expected by the extractor.
   * @param waveform  A 1-D array containing audio samples. For
   *                  models from icefall, the samples should be in the
   *                  range [-1, 1].
   */
  void AcceptWaveform(float sampling_rate, torch::Tensor waveform);

  /** Returns the total number of frames, since the start of the utterance, that
   * are now available.  In an online-decoding context, this will likely
   * increase with time as more data becomes available.
   */
  int32_t NumFramesReady() const;

  /** Returns true if this is the last frame.
   *
   * Frame indices are zero-based, so the first frame is zero.
   */
  bool IsLastFrame(int32_t frame) const;

  /** InputFinished() tells the class you won't be providing any more waveform.
   *
   * It also affects the return value of IsLastFrame().
   */
  void InputFinished();

  /**Get a frame by its index.
   *
   * @param frame  The frame number. It starts from 0.
   *
   * @return Return a 2-D array of shape [1, feature_dim]
   */
  torch::Tensor GetFrame(int32_t frame);

  /**
   * Get the state of the encoder network corresponding to this stream.
   *
   * @return Return the state of the encoder network for this stream.
   */
  torch::IValue GetState() const;

  /**
   * Set the state of the encoder network corresponding to this stream.
   *
   * @param state The state to set.
   */
  void SetState(torch::IValue state);

  /** @TODO(fangjun): Make it an abstract method
   *
   * @param states A list of encoder network states. states[i] is the state
   *               for the i-th stream.
   * @return A state for a batch of streams.
   */
  torch::IValue StackStates(const std::vector<torch::IValue> &states) const;

  /** Inverse operation of StackStates.
   *
   * @param states  State of the encoder network for a batch of streams.
   *
   * @return A list of encoder network states.
   */
  std::vector<torch::IValue> UnStackStates(torch::IValue states) const;

 private:
  class OnlineStreamImpl;
  std::unique_ptr<OnlineStreamImpl> impl_;
};

}  // namespace sherpa

#endif  //  SHERPA_CSRC_ONLINE_STREAM_H_
