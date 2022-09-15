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

#include "sherpa/cpp_api/offline_recognizer.h"

#include <utility>

#include "sherpa/csrc/log.h"
#include "sherpa/csrc/offline_asr.h"
#include "torch/script.h"

namespace sherpa {

class OfflineRecognizer::OfflineRecognizerImpl {
 public:
  OfflineRecognizerImpl(const std::string &nn_model, const std::string &tokens,
                        const DecodingOptions &decoding_opts, bool use_gpu,
                        float sample_rate) {
    OfflineAsrOptions opts;
    opts.nn_model = nn_model;
    opts.tokens = tokens;
    opts.use_gpu = use_gpu;

    switch (decoding_opts.method) {
      case kGreedySearch:
        opts.decoding_method = "greedy_search";
        break;
      case kModifiedBeamSearch:
        opts.decoding_method = "modified_beam_search";
        opts.num_active_paths = decoding_opts.num_active_paths;
        break;
      default:
        SHERPA_LOG(FATAL) << "Unreachable code";
        break;
    }

    // options for bank
    opts.fbank_opts.frame_opts.dither = 0;
    opts.fbank_opts.frame_opts.samp_freq = sample_rate;
    opts.fbank_opts.mel_opts.num_bins = 80;

    asr_ = std::make_unique<OfflineAsr>(opts);
    expected_sample_rate_ = sample_rate;
  }

  std::vector<OfflineRecognitionResult> DecodeFileBatch(
      const std::vector<std::string> &filenames) {
    std::vector<OfflineAsrResult> res =
        asr_->DecodeWaves(filenames, expected_sample_rate_);
    return ToOfflineRecognitionResult(res);
  }

  std::vector<OfflineRecognitionResult> DecodeSamplesBatch(
      const float **samples, const int32_t *samples_length, int32_t n) {
    std::vector<torch::Tensor> tensors(n);
    for (int i = 0; i != n; ++i) {
      auto t = torch::from_blob(const_cast<float *>(samples[i]),
                                {samples_length[i]}, torch::kFloat);
      tensors[i] = std::move(t);
    }
    auto res = asr_->DecodeWaves(tensors);
    return ToOfflineRecognitionResult(res);
  }

  std::vector<OfflineRecognitionResult> DecodeFeaturesBatch(
      const float *features, const int32_t *features_length, int32_t N,
      int32_t T, int32_t C) {
    torch::Tensor tensor = torch::from_blob(const_cast<float *>(features),
                                            {N, T, C}, torch::kFloat);
    torch::Tensor length = torch::from_blob(
        const_cast<int32_t *>(features_length), {N}, torch::kInt);

    auto res = asr_->DecodeFeatures(tensor, length);
    return ToOfflineRecognitionResult(res);
  }

 private:
  std::vector<OfflineRecognitionResult> ToOfflineRecognitionResult(
      const std::vector<OfflineAsrResult> &res) const {
    std::vector<OfflineRecognitionResult> ans(res.size());
    for (size_t i = 0; i != res.size(); ++i) {
      ans[i].text = std::move(res[i].text);
      ans[i].tokens = std::move(res[i].tokens);
      ans[i].timestamps = std::move(res[i].timestamps);
    }
    return ans;
  }

  std::unique_ptr<OfflineAsr> asr_;
  float expected_sample_rate_;
};

OfflineRecognizer::OfflineRecognizer(
    const std::string &nn_model, const std::string &tokens,
    const DecodingOptions &decoding_opts /*= {}*/, bool use_gpu /*=false*/,
    float sample_rate /*= 16000*/)
    : impl_(std::make_unique<OfflineRecognizerImpl>(
          nn_model, tokens, decoding_opts, use_gpu, sample_rate)) {}

OfflineRecognizer::~OfflineRecognizer() = default;

std::vector<OfflineRecognitionResult> OfflineRecognizer::DecodeFileBatch(
    const std::vector<std::string> &filenames) {
  return impl_->DecodeFileBatch(filenames);
}

std::vector<OfflineRecognitionResult> OfflineRecognizer::DecodeSamplesBatch(
    const float **samples, const int32_t *samples_length, int32_t n) {
  return impl_->DecodeSamplesBatch(samples, samples_length, n);
}

std::vector<OfflineRecognitionResult> OfflineRecognizer::DecodeFeaturesBatch(
    const float *features, const int32_t *features_length, int32_t N, int32_t T,
    int32_t C) {
  return impl_->DecodeFeaturesBatch(features, features_length, N, T, C);
}

}  // namespace sherpa
