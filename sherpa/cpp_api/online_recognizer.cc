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

#include "sherpa/cpp_api/online_recognizer.h"

#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_asr.h"

namespace sherpa {

class OnlineRecognizer::OnlineRecognizerImpl {
 public:
  OnlineRecognizerImpl(const std::string &nn_model, const std::string &tokens,
                       const DecodingOptions &decoding_opts, bool use_gpu,
                       float sample_rate)
      : OnlineRecognizerImpl(nn_model, {}, {}, {}, tokens, decoding_opts,
                             use_gpu, sample_rate) {}

  OnlineRecognizerImpl(const std::string &encoder_model,
                       const std::string &decoder_model,
                       const std::string &joiner_model,
                       const std::string &tokens,
                       const DecodingOptions &decoding_opts, bool use_gpu,
                       float sample_rate)
      : OnlineRecognizerImpl({}, encoder_model, decoder_model, joiner_model,
                             tokens, decoding_opts, use_gpu, sample_rate) {}

  // TODO(fangjun): Pass the arguments via a struct
  OnlineRecognizerImpl(const std::string &nn_model,
                       const std::string &encoder_model,
                       const std::string &decoder_model,
                       const std::string &joiner_model,
                       const std::string &tokens,
                       const DecodingOptions &decoding_opts, bool use_gpu,
                       float sample_rate) {
    OnlineAsrOptions opts;
    opts.nn_model = nn_model;
    opts.encoder_model = encoder_model;
    opts.decoder_model = decoder_model;
    opts.joiner_model = joiner_model;

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
    opts.Validate();

    // options for bank
    opts.fbank_opts.frame_opts.dither = 0;
    opts.fbank_opts.frame_opts.samp_freq = sample_rate;
    opts.fbank_opts.mel_opts.num_bins = 80;

    asr_ = std::make_unique<OnlineAsr>(opts);
    expected_sample_rate_ = sample_rate;
  }

  std::unique_ptr<OnlineStream> CreateStream() { return asr_->CreateStream(); }
  bool IsReady(OnlineStream *s) { return asr_->IsReady(s); }

  void DecodeStreams(OnlineStream **ss, int32_t n) {
    asr_->DecodeStreams(ss, n);
  }

  std::string GetResult(OnlineStream *s) const { return asr_->GetResult(s); }

 private:
  std::unique_ptr<OnlineAsr> asr_;
  float expected_sample_rate_;
};

OnlineRecognizer::OnlineRecognizer(
    const std::string &nn_model, const std::string &tokens,
    const DecodingOptions &decoding_opts /*= {}*/, bool use_gpu /*= false*/,
    float sample_rate /*= 16000*/)
    : impl_(std::make_unique<OnlineRecognizerImpl>(
          nn_model, tokens, decoding_opts, use_gpu, sample_rate)) {}

OnlineRecognizer::OnlineRecognizer(
    const std::string &encoder_model, const std::string &decoder_model,
    const std::string &joiner_model, const std::string &tokens,
    const DecodingOptions &decoding_opts /*= {}*/, bool use_gpu /*= false*/,
    float sample_rate /*= 16000*/)
    : impl_(std::make_unique<OnlineRecognizerImpl>(
          encoder_model, decoder_model, joiner_model, tokens, decoding_opts,
          use_gpu, sample_rate)) {}

OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() {
  torch::NoGradGuard no_grad;
  return impl_->CreateStream();
}

bool OnlineRecognizer::IsReady(OnlineStream *s) { return impl_->IsReady(s); }

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) {
  torch::NoGradGuard no_grad;
  impl_->DecodeStreams(ss, n);
}

std::string OnlineRecognizer::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

}  // namespace sherpa
