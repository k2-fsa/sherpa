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

#include "sherpa/csrc/decode_stream.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace sherpa {
DecodeStream::DecodeStream(const RnntConformerModel::State &initial_state,
                           const torch::Tensor &decoder_out,
                           int32_t context_size, int32_t blank_id)
    : context_size_(context_size),
      blank_id_(blank_id),
      state_(initial_state),
      decoder_out_(decoder_out) {
  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.samp_freq = 16000;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.frame_shift_ms = 10.0;
  fbank_opts.frame_opts.frame_length_ms = 25.0;
  fbank_opts.mel_opts.num_bins = 80;

  feature_extractor_ = std::shared_ptr<kaldifeat::OnlineFbank>(
      new kaldifeat::OnlineFbank(fbank_opts));
  hyp_ = {blank_id_, blank_id_};
}

void DecodeStream::AcceptWaveform(const torch::Tensor &waveform,
                                  int32_t sampling_rate) {
  std::lock_guard<std::mutex> lock(feature_mutex_);
  feature_extractor_->AcceptWaveform(sampling_rate, waveform);
  FetchFrames();
}

void DecodeStream::InputFinished() {
  std::lock_guard<std::mutex> lock(feature_mutex_);
  feature_extractor_->InputFinished();
  FetchFrames();
}

void DecodeStream::AddTailPaddings(int32_t n) {
  auto tail_padding = torch::full({1, 80}, log_eps_);
  std::lock_guard<std::mutex> lock(feature_mutex_);
  features_.reserve(features_.size() + n);
  for (int32_t i = 0; i < n; ++i) features_.push_back(tail_padding);
}

bool DecodeStream::IsFinished() /* const */ {
  std::lock_guard<std::mutex> lock(feature_mutex_);
  return feature_extractor_->IsLastFrame(num_fetched_frames_ - 1) &&
         features_.empty();
}

torch::Tensor DecodeStream::GetFeature(int32_t length, int32_t shift) {
  // Wait until there are enough feature frames
  while (features_.size() < length &&
         !feature_extractor_->IsLastFrame(num_fetched_frames_ - 1)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::lock_guard<std::mutex> lock(feature_mutex_);
  auto return_tensor = torch::cat(
      std::vector<torch::Tensor>(
          features_.begin(),
          features_.begin() +
              std::min(length, static_cast<int32_t>(features_.size()))),
      0);

  features_ = std::vector<torch::Tensor>(
      features_.begin() +
          std::min(shift, static_cast<int32_t>(features_.size())),
      features_.end());

  return return_tensor;
}

void DecodeStream::FetchFrames() {
  while (num_fetched_frames_ < feature_extractor_->NumFramesReady()) {
    auto frame = feature_extractor_->GetFrame(num_fetched_frames_);
    features_.push_back(frame);
    num_fetched_frames_ += 1;
  }
}

}  // namespace sherpa
