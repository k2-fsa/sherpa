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

#include "kaldi_native_io/csrc/kaldi-io.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "kaldifeat/csrc/feature-fbank.h"
#include "sentencepiece_processor.h"  // NOLINT
#include "sherpa/csrc/file_utils.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/parse_options.h"
#include "sherpa/csrc/rnnt_beam_search.h"
#include "sherpa/csrc/rnnt_conformer_model.h"
#include "torch/script.h"

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa.

Usage:
  ./bin/sherpa --help

  ./bin/sherpa \
    --nn-model=/path/to/jit_cpu.pt \
    --bpe-model=/path/to/bpe.model \
    --use-gpu=false \
    foo.wav \
)";

struct Options {
  /// Path to torchscript model
  std::string nn_model;

  /// Path to BPE model file. Used only for BPE based models
  std::string bpe_model;

  bool use_gpu = false;

  void Register(sherpa::ParseOptions *po) {
    po->Register("nn-model", &nn_model, "Path to the torchscript model");
    po->Register("bpe-model", &bpe_model,
                 "Path to the BPE model. Used only for BPE based models");
    po->Register(
        "use-gpu", &use_gpu,
        "true to use GPU for computation. false to use CPU.\n"
        "If true, it uses the first device. You can use the environment "
        "variable CUDA_VISIBLE_DEVICES to select which device to use.");
  }

  void Validate() const {
    if (nn_model.empty()) {
      SHERPA_LOG(FATAL) << "Please provide --nn-model";
    }

    if (!sherpa::FileExists(nn_model)) {
      SHERPA_LOG(FATAL) << "\n--nn-model=" << nn_model << "\n"
                        << nn_model << " does not exist!";
    }

    if (!sherpa::FileExists(bpe_model)) {
      SHERPA_LOG(FATAL) << "\n--bpe-model=" << bpe_model << "\n"
                        << bpe_model << " does not exist!";
    }
  }

  std::string ToString() const {
    std::ostringstream os;
    os << "--nn-model=" << nn_model << "\n";
    if (!bpe_model.empty()) {
      os << "--bpe-model=" << bpe_model << "\n";
    }
    os << "--use-gpu=" << std::boolalpha << use_gpu << "\n";
    return os.str();
  }
};

static void RegisterFrameExtractionOptions(
    sherpa::ParseOptions *po, kaldifeat::FrameExtractionOptions *opts);

static void RegisterMelBanksOptions(sherpa::ParseOptions *po,
                                    kaldifeat::MelBanksOptions *opts);

/** Read wave samples from a file.
 *
 * If the file has multiple channels, only the first channel is returned.
 * Samples are normalized to the range [-1, 1).
 *
 * @param filename Path to the wave file.
 * @param expected_sample_rate  Expected sample rate of the wave file. It aborts
 *                              if the sample rate of the given file is not
 *                              equal to this value.
 *
 * @return Return a pair containing
 *  - A 1-D torch.float32 tensor containing entries in the range [-1, 1)
 *  - The duration in seconds of the wave file.
 */
static std::pair<torch::Tensor, float> ReadWave(const std::string &filename,
                                                float expected_sample_rate);

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  // All models in icefall use training data of sample rate 16000
  float expected_sample_rate = 16000;

  sherpa::ParseOptions po(kUsageMessage);

  Options opts;
  opts.Register(&po);

  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  RegisterFrameExtractionOptions(&po, &fbank_opts.frame_opts);

  fbank_opts.mel_opts.num_bins = 80;
  RegisterMelBanksOptions(&po, &fbank_opts.mel_opts);

  po.Read(argc, argv);
  opts.Validate();
  SHERPA_CHECK_EQ(fbank_opts.frame_opts.samp_freq, expected_sample_rate)
      << "The model was trained using data of sample rate 16000. "
      << "We don't support resample yet";

  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  torch::Device device("cpu");
  if (opts.use_gpu) {
    // You can use the environment CUDA_VISIBLE_DEVICES to control
    // which device is mapped to device 0.
    device = torch::Device("cuda:0");
  }

  SHERPA_LOG(INFO) << "Device: " << device.str();

  sherpa::RnntConformerModel model(opts.nn_model, device);

  fbank_opts.device = device;
  kaldifeat::Fbank fbank(fbank_opts);

  torch::Tensor wave;  // 1-d torch.float32 tensor
  float duration;      // in seconds
  std::tie(wave, duration) = ReadWave(po.GetArg(1), expected_sample_rate);

  torch::Tensor feature = fbank.ComputeFeatures(wave, /*vtln_warp*/ 1.0f);
  torch::Tensor feature_length = torch::tensor({feature.size(0)}, device);

  torch::Tensor encoder_out;
  torch::Tensor encoder_out_length;
  std::tie(encoder_out, encoder_out_length) =
      model.ForwardEncoder(feature.unsqueeze(0), feature_length);

  std::vector<int32_t> hyp =
      sherpa::GreedySearch(model, encoder_out, encoder_out_length)[0];

  sentencepiece::SentencePieceProcessor processor;
  auto status = processor.Load(opts.bpe_model);
  SHERPA_CHECK(status.ok()) << status.ToString();

  std::string text;
  status = processor.Decode(hyp, &text);
  SHERPA_CHECK(status.ok()) << status.ToString();

  SHERPA_LOG(INFO) << text;

  return 0;
}

static void RegisterFrameExtractionOptions(
    sherpa::ParseOptions *po, kaldifeat::FrameExtractionOptions *opts) {
  po->Register("sample-frequency", &opts->samp_freq,
               "Waveform data sample frequency (must match the waveform file, "
               "if specified there)");

  po->Register("frame-length", &opts->frame_length_ms,
               "Frame length in milliseconds");

  po->Register("frame-shift", &opts->frame_shift_ms,
               "Frame shift in milliseconds");

  po->Register(
      "dither", &opts->dither,
      "Dithering constant (0.0 means no dither). "
      "Caution: Samples are normalized to the range [-1, 1). "
      "Please select a small value for dither if you want to enable it");
}

static void RegisterMelBanksOptions(sherpa::ParseOptions *po,
                                    kaldifeat::MelBanksOptions *opts) {
  po->Register("num-mel-bins", &opts->num_bins,
               "Number of triangular mel-frequency bins");
}

/* Read wave samples from a file. Samples are divided by 32768 so that
 * they are normalized to the range [-1, 1). All models from icefall
 * use normalized samples while extracting features..
 */
static std::pair<torch::Tensor, float> ReadWave(const std::string &filename,
                                                float expected_sample_rate) {
  bool binary = true;
  kaldiio::Input ki(filename, &binary);
  kaldiio::WaveHolder wh;
  if (!wh.Read(ki.Stream())) {
    SHERPA_LOG(FATAL) << "Failed to read " << filename;
  }

  auto &wave_data = wh.Value();
  if (wave_data.SampFreq() != expected_sample_rate) {
    SHERPA_LOG(FATAL) << filename << "is expected to have sample rate "
                      << expected_sample_rate << ". Given "
                      << wave_data.SampFreq();
  }

  auto &d = wave_data.Data();

  if (d.NumRows() > 1) {
    SHERPA_LOG(WARNING) << "Only the first channel from " << filename
                        << " is used";
  }

  auto tensor = torch::from_blob(const_cast<float *>(d.RowData(0)),
                                 {d.NumCols()}, torch::kFloat);

  return {tensor / 32768, wave_data.Duration()};
}
