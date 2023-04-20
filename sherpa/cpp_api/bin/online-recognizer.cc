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

#include "kaldi_native_io/csrc/kaldi-table.h"
#include "kaldi_native_io/csrc/text-utils.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "sherpa/cpp_api/online-stream.h"
#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/fbank-features.h"
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
    --decoding-method=greedy_search
    foo.wav \
    bar.wav

To use fast_beam_search with an LG, use

  ./bin/sherpa-online \
    --decoding-method=fast_beam_search \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --lg=/path/to/LG.pt \
    --use-gpu=false \
    foo.wav \
    bar.wav

(3) To use an LSTM model for recognition

  ./bin/sherpa-online \
    --encoder-model=/path/to/encoder_jit_trace.pt \
    --decoder-model=/path/to/decoder_jit_trace.pt \
    --joiner-model=/path/to/joiner_jit_trace.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    foo.wav \
    bar.wav

(4) To use a streaming Zipformer model for recognition

  ./bin/sherpa-online \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    foo.wav \
    bar.wav

(5) To decode wav.scp

  ./bin/sherpa-online \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --use-wav-scp=true \
    scp:wav.scp \
    ark,scp,t:result.ark,result.scp

See
https://k2-fsa.github.io/sherpa/cpp/pretrained_models/online_transducer.html
for more details.
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
  bool use_wav_scp = false;  // true to use wav.scp as input

  // Number of seconds for tail padding
  float padding_seconds = 0.8;

  sherpa::ParseOptions po(kUsageMessage);

  po.Register("use-wav-scp", &use_wav_scp,
              "If true, user should provide two arguments: "
              "scp:wav.scp ark,scp,t:results.ark,results.scp");

  po.Register("padding-seconds", &padding_seconds,
              "Number of seconds for tail padding.");

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

  SHERPA_CHECK_GE(padding_seconds, 0);

  SHERPA_LOG(INFO) << "decoding method: " << config.decoding_method;

  torch::Tensor tail_padding = torch::zeros(
      {static_cast<int32_t>(padding_seconds * expected_sample_rate)},
      torch::kFloat);

  sherpa::OnlineRecognizer recognizer(config);
  if (use_wav_scp) {
    SHERPA_CHECK_EQ(po.NumArgs(), 2)
        << "Please use something like:\n"
        << "scp:wav.scp ark,scp,t:results.scp,results.ark\n"
        << "if you provide --use-wav-scp=true";

    if (kaldiio::ClassifyRspecifier(po.GetArg(1), nullptr, nullptr) ==
        kaldiio::kNoRspecifier) {
      SHERPA_LOG(FATAL) << "Please provide an rspecifier. Current value is: "
                        << po.GetArg(1);
    }

    if (kaldiio::ClassifyWspecifier(po.GetArg(2), nullptr, nullptr, nullptr) ==
        kaldiio::kNoWspecifier) {
      SHERPA_LOG(FATAL) << "Please provide a wspecifier. Current value is: "
                        << po.GetArg(2);
    }

    kaldiio::TableWriter<kaldiio::TokenVectorHolder> writer(po.GetArg(2));

    kaldiio::SequentialTableReader<kaldiio::WaveHolder> wav_reader(
        po.GetArg(1));

    int32_t num_decoded = 0;
    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string key = wav_reader.Key();
      SHERPA_LOG(INFO) << "\n" << num_decoded++ << ": decoding " << key;
      auto &wave_data = wav_reader.Value();
      if (wave_data.SampFreq() != expected_sample_rate) {
        SHERPA_LOG(FATAL) << wav_reader.Key()
                          << "is expected to have sample rate "
                          << expected_sample_rate << ". Given "
                          << wave_data.SampFreq();
      }
      auto &d = wave_data.Data();
      if (d.NumRows() > 1) {
        SHERPA_LOG(WARNING)
            << "Only the first channel from " << wav_reader.Key() << " is used";
      }

      auto tensor = torch::from_blob(const_cast<float *>(d.RowData(0)),
                                     {d.NumCols()}, torch::kFloat) /
                    32768;
      auto s = recognizer.CreateStream();
      s->AcceptWaveform(expected_sample_rate, tensor);
      s->AcceptWaveform(expected_sample_rate, tail_padding);
      s->InputFinished();

      while (recognizer.IsReady(s.get())) {
        recognizer.DecodeStream(s.get());
      }
      auto result = recognizer.GetResult(s.get());

      SHERPA_LOG(INFO) << "\nresult: " << result.text;

      std::vector<std::string> words;
      kaldiio::SplitStringToVector(result.text, " ", true, &words);
      writer.Write(key, words);
    }
  } else {
    int32_t num_waves = po.NumArgs();
    if (num_waves == 1) {
      // simulate streaming
      torch::Tensor wave =
          sherpa::ReadWave(po.GetArg(1), expected_sample_rate).first;

      auto s = recognizer.CreateStream();

      int32_t chunk = 0.2 * expected_sample_rate;
      int32_t num_samples = wave.numel();

      std::string last;
      for (int32_t start = 0; start < num_samples;) {
        int32_t end = std::min(start + chunk, num_samples);
        torch::Tensor samples =
            wave.index({torch::indexing::Slice(start, end)});
        start = end;

        s->AcceptWaveform(expected_sample_rate, samples);

        while (recognizer.IsReady(s.get())) {
          recognizer.DecodeStream(s.get());
        }

        auto r = recognizer.GetResult(s.get());

        if (!r.text.empty() && r.text != last) {
          last = r.text;
          std::cout << r.AsJsonString() << "\n";
        }
      }

      s->AcceptWaveform(expected_sample_rate, tail_padding);
      s->InputFinished();
      while (recognizer.IsReady(s.get())) {
        recognizer.DecodeStream(s.get());
      }
      auto r = recognizer.GetResult(s.get());

      if (!r.text.empty() && r.text != last) {
        last = r.text;
        std::cout << r.AsJsonString() << ", size: " << r.text.size() << "\n";
      }
    } else {
      // For multiple waves, we don't use simulate streaming since
      // it would complicate the code. Please use
      // sherpa-online-websocket-server and
      // sherpa-online-websocket-client and for that.
      std::vector<std::unique_ptr<sherpa::OnlineStream>> ss;
      std::vector<sherpa::OnlineStream *> p_ss;

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
        auto r = recognizer.GetResult(p_ss[i - 1]);
        os << r.text << "\n";
        os << r.AsJsonString() << "\n\n";
      }

      std::cerr << os.str();
    }
  }

  return 0;
}
