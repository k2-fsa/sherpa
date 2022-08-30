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

  SHERPA_CHECK_EQ(po.NumArgs(), 1)
      << "We support only decoding 1 wave right now";
  std::string wave_filename = po.GetArg(1);
  SHERPA_LOG(INFO) << "Decoding " << wave_filename;
  float sampling_rate = opts.fbank_opts.frame_opts.samp_freq;

  torch::Tensor wave = sherpa::ReadWave(wave_filename, sampling_rate).first;

  sherpa::OnlineAsr online_asr(opts);

  auto s = online_asr.CreateStream();

  int32_t num_samples = wave.numel();
  int32_t k = 1600;  // feed this number of samples each time
  for (int32_t c = 0; c < num_samples; c += k) {
    int32_t start = c;
    int32_t end = std::min(c + k, num_samples);
    s->AcceptWaveform(sampling_rate, wave.slice(/*dim*/ 0, start, end));
    if (online_asr.IsReady(s.get())) {
      online_asr.DecodeStream(s.get());
    }
  }

  s->InputFinished();
  // TODO(fangjun): Handle the remaining frames

  std::cout << "results:\n" << online_asr.GetResults(s.get()) << "\n";
}
