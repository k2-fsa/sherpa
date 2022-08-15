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
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/offline_asr.h"
#include "sherpa/csrc/parse_options.h"
#include "torch/script.h"

static constexpr const char *kUsageMessage = R"(
Offline (non-streaming) automatic speech recognition with sherpa.

Usage:
(1) View help information.

  ./bin/sherpa --help

(2) Use a BPE-base model for recognition

  ./bin/sherpa \
    --nn-model=/path/to/cpu_jit.pt \
    --bpe-model=/path/to/bpe.model \
    --use-gpu=false \
    foo.wav \
    bar.wav

Note: You can get a pre-trained model for testing by visiting
https://huggingface.co/wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2/tree/main/exp

(3) Use a non-BPE-based model for recognition

  ./bin/sherpa \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    foo.wav \
    bar.wav

Note: You can get a pre-trained model for testing by visiting
https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2/tree/main/exp
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

  // All models in icefall use training data with sample rate 16000
  float expected_sample_rate = 16000;

  sherpa::ParseOptions po(kUsageMessage);

  sherpa::OfflineAsrOptions opts;
  opts.Register(&po);

  po.Read(argc, argv);
  opts.Validate();

  SHERPA_CHECK_EQ(opts.fbank_opts.frame_opts.samp_freq, expected_sample_rate)
      << "The model was trained using training data with sample rate 16000. "
      << "We don't support resample yet";

  sherpa::OfflineAsr offline_asr(opts);
  SHERPA_LOG(INFO) << "\n" << opts.ToString();

  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (po.NumArgs() == 1) {
    auto result = offline_asr.DecodeWave(po.GetArg(1), expected_sample_rate);

    SHERPA_LOG(INFO) << "\nfilename: " << po.GetArg(1)
                     << "\nresult: " << result.text;
  } else {
    std::vector<std::string> filenames(po.NumArgs());
    for (int i = 1; i <= po.NumArgs(); ++i) {
      filenames[i - 1] = po.GetArg(i);
    }
    auto results = offline_asr.DecodeWaves(filenames, expected_sample_rate);
    std::ostringstream os;
    for (size_t i = 0; i != results.size(); ++i) {
      os << "filename: " << filenames[i] << "\n"
         << "result: " << results[i].text << "\n\n";
    }

    SHERPA_LOG(INFO) << "\n" << os.str();
  }

  return 0;
}
