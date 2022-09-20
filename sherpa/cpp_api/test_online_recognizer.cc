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
#include "sherpa/csrc/fbank_features.h"

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  if (argc < 4) {
    std::cerr << "Usage: ./bin/test_decode_files /path/to/nn_model "
                 "/path/to/tokens.txt foo.wav [bar.wav [foobar.wav] ... ]\n";
    exit(EXIT_FAILURE);
  }
  std::string nn_model = argv[1];
  std::string tokens = argv[2];
  float sample_rate = 16000;
  bool use_gpu = false;

  sherpa::DecodingOptions opts;
  opts.method = sherpa::kGreedySearch;
  sherpa::OnlineRecognizer recognizer(nn_model, tokens, opts, use_gpu,
                                      sample_rate);

  torch::Tensor tail_padding =
      torch::zeros({static_cast<int32_t>(0.4 * sample_rate)}, torch::kFloat);

  std::vector<std::unique_ptr<sherpa::OnlineStream>> streams;
  for (int32_t i = 3; i < argc; ++i) {
    auto s = recognizer.CreateStream();

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
      if (recognizer.IsReady(s.get())) {
        ready_streams.push_back(s.get());
      }
    }

    if (ready_streams.empty()) {
      break;
    }

    recognizer.DecodeStreams(ready_streams.data(), ready_streams.size());
  }

  std::vector<std::string> results;
  for (auto &s : streams) {
    results.push_back(recognizer.GetResult(s.get()));
  }

  std::ostringstream os;
  for (int32_t i = 0; i != results.size(); ++i) {
    os << "filename: " << argv[3 + i] << "\n";
    os << "result: " << results[i] << "\n\n";
  }

  std::cout << os.str();

  return 0;
}
