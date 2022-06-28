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

class DecodeStream {
 public:
  DecodeStream(const RnntConformerModel::State &initial_state,
               const torch::Tensor &decoder_out, int32_t context_size = 2,
               int32_t blank_id = 0);

  void AcceptWaveform(const torch::Tensor &waveform,
                      int32_t sampling_rate = 16000);

  void InputFinished();

  void AddTailPaddings(int32_t n = 20);

  torch::Tensor GetFeature(int32_t length, int32_t shift);

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

  bool IsFinished() const {
    std::lock_guard<std::mutex> lock(*feature_mutex_);
    return feature_extractor_->IsLastFrame(num_fetched_frames_) &&
           features_.empty();
  }

  ~DecodeStream() { delete feature_extractor_; }

 private:
  void FetchFrames();

  float log_eps_ = -23.025850929940457f;  // math.log(1e-10)
  int32_t num_fetched_frames_ = 0;
  int32_t num_processed_frames_ = 0;
  int32_t context_size_;
  int32_t blank_id_;
  std::mutex *feature_mutex_;
  std::vector<int32_t> hyp_;
  RnntConformerModel::State state_;
  torch::Tensor decoder_out_;
  std::vector<torch::Tensor> features_;
  kaldifeat::OnlineFbank *feature_extractor_;
};

}  // namespace sherpa
#endif  // SHERPA_ANDROID_SHERPA_DECODE_STREAM_H_
