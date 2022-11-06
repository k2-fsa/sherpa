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
#include "sherpa/cpp_api/offline-recognizer.h"

#include "kaldi_native_io/csrc/kaldi-table.h"
#include "kaldi_native_io/csrc/text-utils.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/log.h"
#include "torch/script.h"

static constexpr const char *kUsageMessage = R"(
Offline (non-streaming) automatic speech recognition with sherpa.

See:

  https://k2-fsa.github.io/sherpa/cpp/offline_asr/api.html

for more details.

Usage:
(1) View help information.

  ./bin/sherpa-offline-recognizer --help

(2) Use a pretrained model for recognition

  ./bin/sherpa-offline-recognizer \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    foo.wav \
    bar.wav

Note: You can get pre-trained models for testing by visiting
 - Chinese: https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2/tree/main/exp
 - English: https://huggingface.co/wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2/tree/main/exp

Hint: In case you only have `data/lang_bpe_500/bpe.model`, you can use
`./scripts/bpe_model_to_tokens.py /path/to/bpe.model > tokens.txt` to generate
`tokens.txt` from `bpe.model`.

(3) Decode wav.scp

  ./bin/sherpa-offline-recognizer \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --use-wav-scp=false \
    scp:wav.scp \
    ark,scp,t:results.ark,results.scp

(4) Decode feats.scp

  ./bin/sherpa-offline-recognizer \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --use-feats-scp=false \
    scp:feats.scp \
    ark,scp,t:results.ark,results.scp

Caution: Models from icefall use normalized audio samples, i.e., samples in
the range [-1, 1), to compute features,
while Kaldi uses samples in the range [-32768, 32767] to compute features.
If you use `feats.scp` from Kaldi with models from icefall, you won't get
expected results.
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
  bool use_wav_scp = false;    // true to use wav.scp as input
  bool use_feats_scp = false;  // true to use feats.scp as input
  int32_t batch_size = 10;

  sherpa::ParseOptions po(kUsageMessage);
  sherpa::OfflineRecognizerConfig config;
  config.Register(&po);

  po.Register("use-wav-scp", &use_wav_scp,
              "If true, user should provide two arguments: "
              "scp:wav.scp ark,scp,t:results.ark,results.scp");

  po.Register("use-feats-scp", &use_feats_scp,
              "If true, user should provide two arguments: "
              "scp:feats.scp ark,scp,t:results.ark,results.scp");

  po.Register("batch-size", &batch_size,
              "Used only when --use-wav-scp=true or --use-feats-scp=true. "
              "It specifies the batch size to use for decoding");

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

  SHERPA_LOG(INFO) << "\n" << config.ToString();

  sherpa::OfflineRecognizer recognizer(config);

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

    SHERPA_CHECK_GT(batch_size, 0);

    kaldiio::TableWriter<kaldiio::TokenVectorHolder> writer(po.GetArg(2));

    kaldiio::SequentialTableReader<kaldiio::WaveHolder> wav_reader(
        po.GetArg(1));

    std::vector<std::string> keys;
    std::vector<torch::Tensor> values;
    std::vector<std::unique_ptr<sherpa::OfflineStream>> ss;
    std::vector<sherpa::OfflineStream *> p_ss;
    for (; !wav_reader.Done(); wav_reader.Next()) {
      keys.push_back(wav_reader.Key());
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
      s->AcceptSamples(tensor.data_ptr<float>(), tensor.numel());
      ss.push_back(std::move(s));
      p_ss.push_back(ss.back().get());

      if (static_cast<int32_t>(keys.size()) >= batch_size) {
        // now for recognition
        recognizer.DecodeStreams(p_ss.data(), p_ss.size());

        for (size_t i = 0; i != keys.size(); ++i) {
          std::vector<std::string> words;
          kaldiio::SplitStringToVector(ss[i]->GetResult().text, " ", true,
                                       &words);
          writer.Write(keys[i], words);
        }
        keys.clear();
        ss.clear();
        p_ss.clear();
      }
    }

    if (!keys.empty()) {
      recognizer.DecodeStreams(p_ss.data(), p_ss.size());
      for (size_t i = 0; i != keys.size(); ++i) {
        std::vector<std::string> words;
        kaldiio::SplitStringToVector(ss[i]->GetResult().text, " ", true,
                                     &words);
        writer.Write(keys[i], words);
      }
      keys.clear();
      ss.clear();
      p_ss.clear();
    }

    return 0;
  }

  if (use_feats_scp) {
    SHERPA_CHECK_EQ(po.NumArgs(), 2)
        << "Please use something like:\n"
        << "scp:feats.scp ark,scp,t:results.scp,results.ark\n"
        << "if you provide --use-feats-scp=true";

    SHERPA_CHECK_EQ(po.NumArgs(), 2)
        << "Please use something like:\n"
        << "scp:feats.scp ark,scp,t:results.scp,results.ark\n"
        << "if you provide --use-feats-scp=true";

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

    SHERPA_CHECK_GT(batch_size, 0);

    kaldiio::TableWriter<kaldiio::TokenVectorHolder> writer(po.GetArg(2));

    kaldiio::SequentialTableReader<
        kaldiio::KaldiObjectHolder<kaldiio::Matrix<float>>>
        feature_reader(po.GetArg(1));
    std::vector<std::string> keys;
    std::vector<torch::Tensor> values;
    std::vector<std::unique_ptr<sherpa::OfflineStream>> ss;
    std::vector<sherpa::OfflineStream *> p_ss;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      keys.push_back(feature_reader.Key());
      auto &d = feature_reader.Value();
      auto tensor = torch::from_blob(const_cast<float *>(d.Data()),
                                     {d.NumRows(), d.NumCols()}, torch::kFloat);
      auto s = recognizer.CreateStream();
      s->AcceptFeatures(tensor.data_ptr<float>(), tensor.size(0),
                        tensor.size(1));
      ss.push_back(std::move(s));
      p_ss.push_back(ss.back().get());

      if (static_cast<int32_t>(keys.size()) >= batch_size) {
        recognizer.DecodeStreams(p_ss.data(), p_ss.size());

        for (size_t i = 0; i != keys.size(); ++i) {
          std::vector<std::string> words;
          kaldiio::SplitStringToVector(ss[i]->GetResult().text, " ", true,
                                       &words);
          writer.Write(keys[i], words);
        }
        keys.clear();
        ss.clear();
        p_ss.clear();
      }
    }

    if (!keys.empty()) {
      recognizer.DecodeStreams(p_ss.data(), p_ss.size());
      for (size_t i = 0; i != keys.size(); ++i) {
        std::vector<std::string> words;
        kaldiio::SplitStringToVector(ss[i]->GetResult().text, " ", true,
                                     &words);
        writer.Write(keys[i], words);
      }
      keys.clear();
      ss.clear();
      p_ss.clear();
    }

    return 0;
  }

  if (po.NumArgs() == 1) {
    auto s = recognizer.CreateStream();
    s->AcceptWaveFile(po.GetArg(1));
    recognizer.DecodeStream(s.get());

    SHERPA_LOG(INFO) << "\nfilename: " << po.GetArg(1)
                     << "\nresult: " << s->GetResult().text;
  } else {
    std::vector<std::unique_ptr<sherpa::OfflineStream>> ss;
    std::vector<sherpa::OfflineStream *> p_ss;
    for (int32_t i = 1; i <= po.NumArgs(); ++i) {
      auto s = recognizer.CreateStream();
      s->AcceptWaveFile(po.GetArg(i));
      ss.push_back(std::move(s));
      p_ss.push_back(ss.back().get());
    }
    recognizer.DecodeStreams(p_ss.data(), p_ss.size());
    std::ostringstream os;
    for (int32_t i = 0; i < po.NumArgs(); ++i) {
      os << "filename: " << po.GetArg(i + 1) << "\n"
         << "result: " << ss[i]->GetResult().text << "\n\n";
    }

    SHERPA_LOG(INFO) << "\n" << os.str();
  }

  return 0;
}
