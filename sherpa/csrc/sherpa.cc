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
#include "kaldi_native_io/csrc/kaldi-table.h"
#include "kaldi_native_io/csrc/text-utils.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/offline_asr.h"
#include "sherpa/csrc/parse_options.h"
#include "torch/script.h"

static constexpr const char *kUsageMessage = R"(
Offline (non-streaming) automatic speech recognition with sherpa.

Usage:
(1) View help information.

  ./bin/sherpa --help

(2) Use a pretrained model for recognition

  ./bin/sherpa \
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

  ./bin/sherpa \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --use-wav-scp=false \
    scp:wav.scp \
    ark,scp,t:results.ark,results.scp

(4) Decode feats.scp

  ./bin/sherpa \
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

  sherpa::OfflineAsrOptions opts;
  opts.Register(&po);

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

  opts.Validate();

  SHERPA_CHECK_EQ(opts.fbank_opts.frame_opts.samp_freq, expected_sample_rate)
      << "The model was trained using training data with sample rate 16000. "
      << "We don't support resample yet";

  sherpa::OfflineAsr offline_asr(opts);
  SHERPA_LOG(INFO) << "\n" << opts.ToString();

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
                                     {d.NumCols()}, torch::kFloat);
      values.push_back(tensor / 32768);

      if (keys.size() >= batch_size) {
        // now for recognition
        auto results = offline_asr.DecodeWaves(values);
        for (size_t i = 0; i != keys.size(); ++i) {
          std::vector<std::string> words;
          kaldiio::SplitStringToVector(results[i].text, " ", true, &words);
          writer.Write(keys[i], words);
        }
        keys.clear();
        values.clear();
      }
    }

    if (!keys.empty()) {
      auto results = offline_asr.DecodeWaves(values);
      for (size_t i = 0; i != keys.size(); ++i) {
        std::vector<std::string> words;
        kaldiio::SplitStringToVector(results[i].text, " ", true, &words);
        writer.Write(keys[i], words);
      }
      keys.clear();
      values.clear();
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
    for (; !feature_reader.Done(); feature_reader.Next()) {
      keys.push_back(feature_reader.Key());
      auto &d = feature_reader.Value();
      auto tensor = torch::from_blob(const_cast<float *>(d.Data()),
                                     {d.NumRows(), d.NumCols()}, torch::kFloat);
      values.push_back(tensor.clone());
      if (keys.size() >= batch_size) {
        // now for recognition
        auto results = offline_asr.DecodeFeatures(values);
        for (size_t i = 0; i != keys.size(); ++i) {
          std::vector<std::string> words;
          kaldiio::SplitStringToVector(results[i].text, " ", true, &words);
          writer.Write(keys[i], words);
        }
        keys.clear();
        values.clear();
      }
    }

    if (!keys.empty()) {
      auto results = offline_asr.DecodeFeatures(values);
      for (size_t i = 0; i != keys.size(); ++i) {
        std::vector<std::string> words;
        kaldiio::SplitStringToVector(results[i].text, " ", true, &words);
        writer.Write(keys[i], words);
      }
      keys.clear();
      values.clear();
    }

    return 0;
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
