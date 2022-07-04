/**
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
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

#ifndef SHERPA_ANDROID_SHERPA_DECODE_STREAM_H_
#define SHERPA_ANDROID_SHERPA_DECODE_STREAM_H_

#include <mutex>

#include "kaldifeat/csrc/online-feature.h"
#include "sherpa/csrc/rnnt_conformer_model.h"
#include "torch/script.h"

namespace sherpa {

// Each sequence will have a `DecodeStream` which caches the internal decoding
// states. During the decoding, the recorder sends waveforms to DecodeStream,
// the DecodeStream converts this waveforms into features (mel-bank), then the
// decoder fetches features from DecodeStream to do decoding and store back the
// internal decoder states into DecodeStream.
class DecodeStream {
 public:
  /* Constructor.
   *
   * @param initial_state Initial decode states of the model, e.g. the return
   *                      value of `GetEncoderInitStates` in RnntModel.
   * @param decoder_out The output of decoder network, a 2-D tensor of shape
   *                    (N, C).
   * @param context_size The context_size in stateless decoder network.
   * @param blank_id Blank token id of modeling units.
   */
  DecodeStream(const RnntConformerModel::State &initial_state,
               const torch::Tensor &decoder_out, int32_t context_size = 2,
               int32_t blank_id = 0);

  /* Feed audio samples to the feature extractor and compute features
   * if there are enough samples available.
   *
   * Caution:
   *   The range of the audio samples should match the one used in the
   *   training. That is, if you use the range [-1, 1] in the training, then
   *   the input audio samples should also be normalized to [-1, 1].
   *
   * @param waveform A 1-D torch tensor of dtype torch.float32 containing audio
   *                   samples. It should be on CPU.
   *
   * @param sampling_rate The sampling rate of the input audio samples. It is
   *                      used for sanity check to ensure that the input
   *                      sampling rate equals to the one used in the extractor.
   *                      If they are not equal, then no resampling will be
   *                      performed; instead an error will be thrown.
   */
  void AcceptWaveform(const torch::Tensor &waveform,
                      int32_t sampling_rate = 16000);

  /* Signal that no more audio samples available and the feature
   * extractor should flush the buffered samples to compute frames.
   */
  void InputFinished();

  /* Add some tail paddings so that we have enough context to process
   * frames at the very end of an utterance.
   *
   * @param n Number of tail padding frames to be added. You can increase it if
   *          it happens that there are many missing tokens for the last word of
   *          an utterance.
   */
  void AddTailPaddings(int32_t n = 20);

  /* Get the feature frames according to the given length and update the
   * `features_` data member according to the given shift.
   *
   * Note: It will wait until there are enough frames to return, only if the
   *       `InputFinished` is signaled, it always returns the features of the
   *       given length.
   *
   * @param length The length of the return feature frames.
   * @param shift The `features_` would be updated with
   *              `features_ = features_[shift:]`.
   * @return A tensor constructed with `torch.cat(features_[0: length], dim=0)`.
   */
  torch::Tensor GetFeature(int32_t length, int32_t shift);

  /* See docs in data member `num_processed_frames_` for details of what is
   * `num_processed_frames_`.
   */
  int32_t GetNumProcessedFrames() const { return num_processed_frames_; }
  void UpdateNumProcessedFrames(int32_t processed_frames) {
    num_processed_frames_ += processed_frames;
  }

  torch::Tensor GetDecoderOut() const { return decoder_out_; }
  void SetDecoderOut(torch::Tensor decoder_out) { decoder_out_ = decoder_out; }

  RnntConformerModel::State GetState() const { return state_; }
  void SetState(RnntConformerModel::State &state) { state_ = state; }

  std::vector<int32_t> GetHyp() const { return hyp_; }
  void SetHyp(std::vector<int32_t> &hyp) { hyp_ = hyp; }

  int32_t ContextSize() const { return context_size_; }
  int32_t BlankId() const { return blank_id_; }

  /* Return true if all the feature frames in this stream are decoded, otherwise
   * false.
   */
  bool IsFinished() /* const */;

  ~DecodeStream() {}

 private:
  // Fetch frames from the feature extractor.
  void FetchFrames();

  float log_eps_ = -23.025850929940457f;  // math.log(1e-10)
  int32_t num_fetched_frames_ = 0;    // A counter indicating how many frames of
                                      // waveforms we have converted into
                                      // features (Note: The frame indexes are
                                      // before subsampling)
  int32_t num_processed_frames_ = 0;  // A counter indicating how many frames of
                                      // features we have decoded. (Note: The
                                      // frame indexes are after subsampling).
  int32_t context_size_;
  int32_t blank_id_;
  std::mutex feature_mutex_;  // mutex to protect feature_extractor_ and
                              // features_, as both recorder thread and decoder
                              // thread will modify them.
  std::vector<int32_t> hyp_;  // Partial or final decode results.
  RnntConformerModel::State state_;  // Internal states (i.e. caches) of
                                     // conformer model.
  torch::Tensor decoder_out_;        // The decoder output of decoder network,
                                     // a tensor of shape (1, C).
  std::vector<torch::Tensor> features_;  // A container containing extracted
                                         // features.
  std::shared_ptr<kaldifeat::OnlineFbank> feature_extractor_;
};

}  // namespace sherpa
#endif  // SHERPA_ANDROID_SHERPA_DECODE_STREAM_H_
