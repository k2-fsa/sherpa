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

#include "sherpa/cpp_api/online-recognizer.h"

#include <algorithm>

#include "sherpa/cpp_api/online-stream.h"
#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/fbank_features.h"
#include "sherpa/csrc/log.h"

static constexpr const char *kUsageMessage = R"(
Online (streaming) automatic speech recognition with sherpa.

Usage:
(1) View help information.

  ./bin/sherpa-online --help

(2) Use a pretrained model for recognition

  ./bin/sherpa-online \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    foo.wav \
    bar.wav

Note: You can get pre-trained models for testing by visiting
 - English: https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

(3) To use an LSTM model for recognition

  ./bin/sherpa-online \
    --encoder-model=/path/to/encoder_jit_trace.pt \
    --decoder-model=/path/to/decoder_jit_trace.pt \
    --joiner-model=/path/to/joiner_jit_trace.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    foo.wav \
    bar.wav
)";

int32_t main(int32_t argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  // All models in icefall use training data with sample rate 16000
  float expected_sample_rate = 16000;

  sherpa::ParseOptions po(kUsageMessage);

  sherpa::OnlineRecognizerConfig config;
  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  config.Validate();

  SHERPA_CHECK_EQ(config.feat_config.fbank_opts.frame_opts.samp_freq,
                  expected_sample_rate)
      << "The model was trained using training data with sample rate 16000. "
      << "We don't support resample yet";

  SHERPA_CHECK_GE(po.NumArgs(), 1);

  SHERPA_LOG(INFO) << "decoding method: " << config.decoding_method;

  sherpa::OnlineRecognizer recognizer(config);
  int32_t num_waves = po.NumArgs();
  std::vector<std::unique_ptr<sherpa::OnlineStream>> ss;
  std::vector<sherpa::OnlineStream *> p_ss;

  torch::Tensor tail_padding = torch::zeros(
      {static_cast<int32_t>(0.4 * expected_sample_rate)}, torch::kFloat);

  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    auto s = recognizer.CreateStream();

    torch::Tensor wave =
        sherpa::ReadWave(po.GetArg(i), expected_sample_rate).first;

    s->AcceptWaveform(expected_sample_rate, wave);

    s->AcceptWaveform(expected_sample_rate, tail_padding);
    s->InputFinished();
    ss.push_back(std::move(s));
    p_ss.push_back(ss.back().get());
  }

  std::vector<sherpa::OnlineStream *> ready_streams;
  for (;;) {
    ready_streams.clear();
    for (auto s : p_ss) {
      if (recognizer.IsReady(s)) {
        ready_streams.push_back(s);
      }
    }

    if (ready_streams.empty()) {
      break;
    }
    recognizer.DecodeStreams(ready_streams.data(), ready_streams.size());
  }

  std::ostringstream os;
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    os << po.GetArg(i) << "\n";
    os << recognizer.GetResult(p_ss[i - 1]).text << "\n\n";
  }

  std::cerr << os.str();

  return 0;
}
