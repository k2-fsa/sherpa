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

#include "sherpa/csrc/fbank_features.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_asr.h"
#include "sherpa/csrc/online_stream.h"
#include "sherpa/csrc/parse_options.h"

static constexpr const char *kUsageMessage = R"(
TO DO
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

  sherpa::ParseOptions po(kUsageMessage);

  sherpa::OnlineAsrOptions opts;
  opts.Register(&po);
  po.Read(argc, argv);
  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  opts.Validate();

  SHERPA_CHECK_GE(po.NumArgs(), 1);

  SHERPA_LOG(INFO) << "decoding method: " << opts.decoding_method;

  sherpa::OnlineAsr online_asr(opts);

  float sampling_rate = opts.fbank_opts.frame_opts.samp_freq;

  std::vector<std::unique_ptr<sherpa::OnlineStream>> streams;
  torch::Tensor tail_padding =
      torch::zeros({static_cast<int32_t>(0.4 * sampling_rate)}, torch::kFloat);
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    std::string wave_filename = po.GetArg(i);
    torch::Tensor wave = sherpa::ReadWave(wave_filename, sampling_rate).first;
    auto s = online_asr.CreateStream();
    s->AcceptWaveform(sampling_rate, wave);
    s->AcceptWaveform(sampling_rate, tail_padding);
    s->InputFinished();
    streams.push_back(std::move(s));
  }

  std::vector<sherpa::OnlineStream *> ready_streams;
  while (true) {
    ready_streams.clear();
    for (auto &s : streams) {
      if (online_asr.IsReady(s.get())) {
        ready_streams.push_back(s.get());
      }
    }
    if (ready_streams.empty()) {
      break;
    }
    online_asr.DecodeStreams(ready_streams.data(), ready_streams.size());
  }

  std::ostringstream os;
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    os << "wave_filename: " << po.GetArg(i) << "\n";
    os << "results: " << online_asr.GetResults(streams[i - 1].get()) << "\n\n";
  }
  std::cout << os.str();
}
