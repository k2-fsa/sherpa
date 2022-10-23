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

#include "sherpa/cpp_api/online_stream.h"

#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

#include "kaldifeat/csrc/feature-fbank.h"
#include "kaldifeat/csrc/online-feature.h"
#include "sherpa/csrc/endpoint.h"
#include "sherpa/csrc/hypothesis.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

class OnlineStream::OnlineStreamImpl {
 public:
  OnlineStreamImpl(const EndpointConfig &endpoint_config, float sampling_rate,
                   int32_t feature_dim, int32_t max_feature_vectors) {
    kaldifeat::FbankOptions opts;

    opts.frame_opts.samp_freq = sampling_rate;
    opts.frame_opts.dither = 0;
    opts.frame_opts.snip_edges = false;
    opts.frame_opts.max_feature_vectors = max_feature_vectors;
    opts.mel_opts.num_bins = feature_dim;

    fbank_ = std::make_unique<kaldifeat::OnlineFbank>(opts);
    frame_shift_ms_ = opts.frame_opts.frame_shift_ms;
    endpoint_ = std::make_unique<Endpoint>(endpoint_config);
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

  bool IsEndpoint() const {
    return endpoint_->IsEndpoint(
        num_processed_frames_,
        num_trailing_blank_frames_ * 4,  // subsample factor is 4
        frame_shift_ms_ / 1000.0);
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

  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

  std::vector<int32_t> &GetHyps() { return hyps_; }

  torch::Tensor &GetDecoderOut() { return decoder_out_; }

  Hypotheses &GetHypotheses() { return hypotheses_; }

  int32_t &GetNumTrailingBlankFrames() { return num_trailing_blank_frames_; }

 private:
  std::unique_ptr<kaldifeat::OnlineFbank> fbank_;
  std::unique_ptr<Endpoint> endpoint_;
  mutable std::mutex feat_mutex_;

  torch::IValue state_;
  std::vector<int32_t> hyps_;
  Hypotheses hypotheses_;
  torch::Tensor decoder_out_;
  int32_t num_processed_frames_ = 0;       // before subsampling
  int32_t num_trailing_blank_frames_ = 0;  // after subsampling
  int32_t frame_shift_ms_ = 10;            // before subsampling
};

OnlineStream::OnlineStream(const EndpointConfig &endpoint_config,
                           float sampling_rate, int32_t feature_dim,
                           int32_t max_feature_vectors /*= -1*/)
    : impl_(std::make_unique<OnlineStreamImpl>(
          endpoint_config, sampling_rate, feature_dim, max_feature_vectors)) {}

OnlineStream::~OnlineStream() = default;

void OnlineStream::AcceptWaveform(float sampling_rate, torch::Tensor waveform) {
  impl_->AcceptWaveform(sampling_rate, waveform);
}

int32_t OnlineStream::NumFramesReady() const { return impl_->NumFramesReady(); }

bool OnlineStream::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

bool OnlineStream::IsEndpoint() const { return impl_->IsEndpoint(); }

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

}  // namespace sherpa
