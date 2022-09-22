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
#include "sherpa/csrc/hypothesis.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

class OnlineStream::OnlineStreamImpl {
 public:
  OnlineStreamImpl(float sampling_rate, int32_t feature_dim,
                   int32_t max_feature_vectors) {
    kaldifeat::FbankOptions opts;

    opts.frame_opts.samp_freq = sampling_rate;
    opts.frame_opts.dither = 0;
    opts.frame_opts.snip_edges = false;
    opts.frame_opts.max_feature_vectors = max_feature_vectors;
    opts.mel_opts.num_bins = feature_dim;

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

  torch::IValue StackStates(const std::vector<torch::IValue> &states) const {
    int32_t batch_size = states.size();

    // attn_caches.size() == num_layers
    std::vector<std::vector<std::vector<torch::Tensor>>> attn_caches;
    // We will call torch.stack(attn_caches[i][j]) later

    // conv_caches.size() == num_layers
    std::vector<std::vector<torch::Tensor>> conv_caches;
    // we will call torch.stack(conv_caches[i]) later
    int32_t num_layers = 0;

    for (auto &s : states) {
      // s is a Tuple
      // s[0] contains attn_caches : List[List[torch.Tensor]]
      // s[1] contains conv_caches: List[torch.Tensor]
      //
      // len(attn_caches) == num_layers == len(conv_caches)
      //
      // len(attn_caches[i]) == 3
      // attn_caches[i][0] is a 2-D tensor of shape [memory_size, d_mode]
      // attn_caches[i][1] and attn_caches[i][2] are 2-D tensors of shape
      // [context_size, d_mode]
      auto tuple_ptr = s.toTuple();
      torch::List<torch::IValue> list_attn = tuple_ptr->elements()[0].toList();
      torch::List<torch::IValue> list_conv = tuple_ptr->elements()[1].toList();

      // attn.size() == num_layers
      torch::List<torch::List<torch::Tensor>> attn =
          c10::impl::toTypedList<torch::List<torch::Tensor>>(list_attn);

      torch::List<torch::Tensor> conv =
          c10::impl::toTypedList<torch::Tensor>(list_conv);

      num_layers = attn.size();

      if (attn_caches.empty()) {
        attn_caches.resize(num_layers);
        conv_caches.resize(num_layers);
      }

      for (int32_t l = 0; l != num_layers; ++l) {
        const torch::List<torch::Tensor> &attn_l = attn[l];
        int32_t num_states_this_layer = attn_l.size();

        auto &attn_caches_l = attn_caches[l];
        if (attn_caches_l.empty()) {
          attn_caches_l.resize(num_states_this_layer);
        }

        for (int32_t k = 0; k != num_states_this_layer; ++k) {
          attn_caches_l[k].push_back(attn_l[k]);
        }

        conv_caches[l].push_back(conv[l]);
      }  // for (int32_t l = 0; l != num_layers; ++l)
    }    // for (auto &s : states)

    std::vector<std::vector<torch::Tensor>> stacked_attn_caches(num_layers);
    std::vector<torch::Tensor> stacked_conv_caches(num_layers);

    for (int32_t l = 0; l != num_layers; ++l) {
      auto &attn_caches_l = attn_caches[l];
      auto &stacked_attn_caches_l = stacked_attn_caches[l];
      for (int32_t i = 0; i != attn_caches_l.size(); ++i) {
        stacked_attn_caches_l.push_back(
            torch::stack(attn_caches_l[i], /*dim*/ 1));
      }

      stacked_conv_caches[l] = torch::stack(conv_caches[l], /*dim*/ 0);
    }

    return torch::ivalue::Tuple::create(stacked_attn_caches,
                                        stacked_conv_caches);
  }

  std::vector<torch::IValue> UnStackStates(torch::IValue states) const {
    TORCH_CHECK(states.isTuple(), "Expect a tuple. Given ", states.tagKind());

    auto tuple_ptr = states.toTuple();
    torch::List<torch::IValue> list_attn = tuple_ptr->elements()[0].toList();
    torch::List<torch::IValue> list_conv = tuple_ptr->elements()[1].toList();

    torch::List<torch::List<torch::Tensor>> stacked_attn =
        c10::impl::toTypedList<torch::List<torch::Tensor>>(list_attn);

    torch::List<torch::Tensor> stacked_conv =
        c10::impl::toTypedList<torch::Tensor>(list_conv);

    int32_t batch_size =
        static_cast<const torch::Tensor &>(stacked_conv[0]).size(0);
    int32_t num_layers = stacked_conv.size();
    int32_t num_states_per_layer =
        static_cast<const torch::List<torch::Tensor> &>(stacked_attn[0]).size();

    std::vector<std::vector<std::vector<torch::Tensor>>> unstacked_attn(
        batch_size);

    for (auto &v : unstacked_attn) {
      v.resize(num_layers);
    }

    std::vector<std::vector<torch::Tensor>> unstacked_conv(batch_size);

    for (int32_t l = 0; l != num_layers; ++l) {
      const torch::List<torch::Tensor> &stacked_attn_l = stacked_attn[l];
      std::vector<std::vector<torch::Tensor>> layer_states(
          num_states_per_layer);
      for (int32_t k = 0; k != num_states_per_layer; ++k) {
        std::vector<torch::Tensor> s =
            torch::unbind(stacked_attn_l[k], /*dim*/ 1);
        for (int32_t b = 0; b != batch_size; ++b) {
          unstacked_attn[b][l].push_back(std::move(s[b]));
        }
      }  // for (int32_t k = 0; k != num_states_per_layer; ++k)

      auto v = torch::unbind(stacked_conv[l], /*dim*/ 0);
      for (int32_t b = 0; b != batch_size; ++b) {
        unstacked_conv[b].push_back(v[b]);
      }
    }  // for (int32_t l = 0; l != num_layers; ++l)

    std::vector<torch::IValue> ans(batch_size);
    for (int32_t b = 0; b != batch_size; ++b) {
      ans[b] =
          torch::ivalue::Tuple::create(unstacked_attn[b], unstacked_conv[b]);
    }

    return ans;
  }

  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

  std::vector<int32_t> &GetHyps() { return hyps_; }

  torch::Tensor &GetDecoderOut() { return decoder_out_; }

  Hypotheses &GetHypotheses() { return hypotheses_; }

  int32_t &GetNumTrailingBlankFrames() { return num_trailing_blank_frames_; }

 private:
  std::unique_ptr<kaldifeat::OnlineFbank> fbank_;
  mutable std::mutex feat_mutex_;

  torch::IValue state_;
  std::vector<int32_t> hyps_;
  Hypotheses hypotheses_;
  torch::Tensor decoder_out_;
  int32_t num_processed_frames_ = 0;       // before subsampling
  int32_t num_trailing_blank_frames_ = 0;  // after subsampling
};

OnlineStream::OnlineStream(float sampling_rate, int32_t feature_dim,
                           int32_t max_feature_vectors /*= -1*/)
    : impl_(std::make_unique<OnlineStreamImpl>(sampling_rate, feature_dim,
                                               max_feature_vectors)) {}

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

torch::IValue OnlineStream::StackStates(
    const std::vector<torch::IValue> &states) const {
  return impl_->StackStates(states);
}

std::vector<torch::IValue> OnlineStream::UnStackStates(
    torch::IValue states) const {
  return impl_->UnStackStates(states);
}

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
