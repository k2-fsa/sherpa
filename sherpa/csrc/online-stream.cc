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

#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

#include "kaldifeat/csrc/feature-fbank.h"
#include "kaldifeat/csrc/online-feature.h"
#include "sherpa/cpp_api/endpoint.h"
#include "sherpa/cpp_api/online-stream.h"
#include "sherpa/csrc/hypothesis.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online-transducer-decoder.h"

namespace sherpa {

class OnlineStream::OnlineStreamImpl {
 public:
  explicit OnlineStreamImpl(const kaldifeat::FbankOptions &opts) {
    fbank_ = std::make_unique<kaldifeat::OnlineFbank>(opts);
  }

  void AcceptWaveform(float sampling_rate, torch::Tensor waveform) {
    std::lock_guard<std::mutex> lock(feat_mutex_);
    fbank_->AcceptWaveform(sampling_rate, waveform);
  }

  int32_t NumFramesReady() const {
    std::lock_guard<std::mutex> lock(feat_mutex_);
    return fbank_->NumFramesReady();
  }

  bool IsLastFrame(int32_t frame) const {
    std::lock_guard<std::mutex> lock(feat_mutex_);
    return fbank_->IsLastFrame(frame);
  }

  void InputFinished() {
    std::lock_guard<std::mutex> lock(feat_mutex_);
    fbank_->InputFinished();
  }

  torch::Tensor GetFrame(int32_t frame) {
    std::lock_guard<std::mutex> lock(feat_mutex_);
    return fbank_->GetFrame(frame);
  }

  torch::IValue GetState() const { return state_; }

  void SetState(torch::IValue state) { state_ = std::move(state); }

  void SetResult(const OnlineTransducerDecoderResult &r) { r_ = r; }

  const OnlineTransducerDecoderResult &GetResult() const { return r_; }

  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

  std::vector<int32_t> &GetHyps() { return hyps_; }

  torch::Tensor &GetDecoderOut() { return decoder_out_; }

  Hypotheses &GetHypotheses() { return hypotheses_; }

  int32_t &GetNumTrailingBlankFrames() { return num_trailing_blank_frames_; }

  int32_t &GetWavSegment() { return segment_; }

  int32_t &GetStartFrame() { return start_frame_; }

 private:
  std::unique_ptr<kaldifeat::OnlineFbank> fbank_;
  mutable std::mutex feat_mutex_;

  torch::IValue state_;
  std::vector<int32_t> hyps_;
  Hypotheses hypotheses_;
  torch::Tensor decoder_out_;
  int32_t num_processed_frames_ = 0;       // before subsampling
  int32_t num_trailing_blank_frames_ = 0;  // after subsampling
  /// ID of this segment
  int32_t segment_ = 0;

  /// Starting frame of this segment.
  int32_t start_frame_ = 0;
  OnlineTransducerDecoderResult r_;
};

OnlineStream::OnlineStream(const kaldifeat::FbankOptions &opts)
    : impl_(std::make_unique<OnlineStreamImpl>(opts)) {}

OnlineStream::~OnlineStream() = default;

void OnlineStream::AcceptWaveform(float sampling_rate, torch::Tensor waveform) {
  impl_->AcceptWaveform(sampling_rate, waveform);
}

int32_t OnlineStream::NumFramesReady() const { return impl_->NumFramesReady(); }

bool OnlineStream::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

void OnlineStream::InputFinished() { impl_->InputFinished(); }

torch::Tensor OnlineStream::GetFrame(int32_t frame) {
  return impl_->GetFrame(frame);
}

torch::IValue OnlineStream::GetState() const { return impl_->GetState(); }

void OnlineStream::SetState(torch::IValue state) { impl_->SetState(state); }

int32_t &OnlineStream::GetNumProcessedFrames() {
  return impl_->GetNumProcessedFrames();
}

std::vector<int32_t> &OnlineStream::GetHyps() { return impl_->GetHyps(); }

torch::Tensor &OnlineStream::GetDecoderOut() { return impl_->GetDecoderOut(); }

Hypotheses &OnlineStream::GetHypotheses() { return impl_->GetHypotheses(); }

int32_t &OnlineStream::GetNumTrailingBlankFrames() {
  return impl_->GetNumTrailingBlankFrames();
}

int32_t &OnlineStream::GetWavSegment() {
  return impl_->GetWavSegment();
}

int32_t &OnlineStream::GetStartFrame() {
  return impl_->GetStartFrame();
}

void OnlineStream::SetResult(const OnlineTransducerDecoderResult &r) {
  impl_->SetResult(r);
}

const OnlineTransducerDecoderResult &OnlineStream::GetResult() const {
  return impl_->GetResult();
}

}  // namespace sherpa
