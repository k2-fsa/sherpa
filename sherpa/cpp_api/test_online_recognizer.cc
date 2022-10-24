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
#include <iostream>

#include "sherpa/cpp_api/online_recognizer.h"
#include "sherpa/cpp_api/online_stream.h"
#include "sherpa/csrc/fbank_features.h"  // for sherpa::ReadWave()

static constexpr const char *kUsageMessage = R"(
Online (streaming) automatic speech recognition with sherpa.

Usage:
(1) Use a streaming conv-emformer

  ./bin/test_online_recognizer \
    /path/to/nn_model \
    /path/to/tokens.txt \
    foo.wav [bar.wav [foobar.wav] ... ]

(2) Use a streaming LSTM model

  ./bin/test_online_recognizer \
    /path/to/encoder_model \
    /path/to/decoder_model \
    /path/to/joiner_model \
    /path/to/tokens.txt \
    foo.wav [bar.wav [foobar.wav] ... ]
)";

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  std::string nn_model;  // for RnntConvEmformerModel

  std::string encoder_model;  // for RnntLstmModel
  std::string decoder_model;
  std::string joiner_model;

  std::string tokens;

  if (argc < 4) {
    std::cerr << kUsageMessage << "\n";
    exit(EXIT_FAILURE);
  }

  int32_t wave_start_index;
  auto tmp = std::string(argv[2]);
  if (tmp.rfind(".pt", tmp.size() - 3) == tmp.size() - 3) {
    if (argc < 6) {
      std::cerr << kUsageMessage << "\n";
      exit(EXIT_FAILURE);
    }

    // It is an LSTM model
    encoder_model = argv[1];
    decoder_model = std::move(tmp);
    joiner_model = argv[3];
    tokens = argv[4];

    wave_start_index = 5;
  } else {
    nn_model = argv[1];
    tokens = argv[2];
    wave_start_index = 3;
  }

  float sample_rate = 16000;
  bool use_gpu = false;

  sherpa::DecodingOptions opts;
  opts.method = sherpa::kGreedySearch;
  std::unique_ptr<sherpa::OnlineRecognizer> recognizer;
  if (!nn_model.empty()) {
    recognizer = std::make_unique<sherpa::OnlineRecognizer>(
        nn_model, tokens, opts, use_gpu, sample_rate);
  } else {
    recognizer = std::make_unique<sherpa::OnlineRecognizer>(
        encoder_model, decoder_model, joiner_model, tokens, opts, use_gpu,
        sample_rate);
  }

  torch::Tensor tail_padding =
      torch::zeros({static_cast<int32_t>(0.4 * sample_rate)}, torch::kFloat);

  std::vector<std::unique_ptr<sherpa::OnlineStream>> streams;
  for (int32_t i = wave_start_index; i < argc; ++i) {
    auto s = recognizer->CreateStream();

    std::string wave_filename = argv[i];
    torch::Tensor wave = sherpa::ReadWave(wave_filename, sample_rate).first;
    s->AcceptWaveform(sample_rate, wave);
    s->AcceptWaveform(sample_rate, tail_padding);
    s->InputFinished();

    streams.push_back(std::move(s));
  }

  std::vector<sherpa::OnlineStream *> ready_streams;

  while (true) {
    ready_streams.clear();
    for (auto &s : streams) {
      if (recognizer->IsReady(s.get())) {
        ready_streams.push_back(s.get());
      }
    }

    if (ready_streams.empty()) {
      break;
    }

    recognizer->DecodeStreams(ready_streams.data(), ready_streams.size());
  }

  std::vector<std::string> results;
  for (auto &s : streams) {
    results.push_back(recognizer->GetResult(s.get()));
  }

  std::ostringstream os;
  for (size_t i = 0; i != results.size(); ++i) {
    os << "filename: " << argv[wave_start_index + i] << "\n";
    os << "result: " << results[i] << "\n\n";
  }

  std::cout << os.str();

  return 0;
}
