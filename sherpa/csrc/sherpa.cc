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
#include "sherpa/csrc/file_utils.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/parse_options.h"
#include "torch/script.h"

struct Options {
  /// Path to torchscript model
  std::string nn_model;

  /// Path to BPE model file. Used only for BPE based models
  std::string bpe_model;

  bool use_gpu = false;

  // The sample rate of the training data used to train the model.
  float sample_rate = 16000;

  void Register(sherpa::ParseOptions *po) {
    po->Register("nn-model", &nn_model, "Path to the torchscript model");
    po->Register("bpe-model", &bpe_model,
                 "Path to the BPE model. Used only for BPE based models");
    po->Register(
        "use-gpu", &use_gpu,
        "true to use GPU for computation. false to use CPU.\n"
        "If true, it uses the first device. You can use the environment "
        "variable CUDA_VISIBLE_DEVICES to select which device to use.");

    po->Register(
        "sample-rate", &sample_rate,
        "The sample rate of the training data used to train the model. "
        "The sample rate of the wave should match this value.");
  }

  void Validate() const {
    if (nn_model.empty()) {
      SHERPA_LOG(FATAL) << "Please provide --nn-model";
    }

    if (!sherpa::FileExists(nn_model)) {
      SHERPA_LOG(FATAL) << "\n--nn-model=" << nn_model << "\n"
                        << nn_model << " does not exist!";
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

static torch::Tensor ReadWave(const std::string &filename,
                              float expected_sample_rate);

int main(int argc, char *argv[]) {
  const char *usage = R"(Automatic speech recognition with sherpa.
Usage:
  ./bin/sherpa --help

  ./bin/sherpa \
    --nn-model=/path/to/jit_cpu.pt \
    --bpe-model=/path/to/bpe.model \
    --use-gpu=false \
    foo.wav \
    bar.wav

  )";
  sherpa::ParseOptions po(usage);

  Options opts;
  opts.Register(&po);

  po.Read(argc, argv);
  opts.Validate();

  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }
  SHERPA_LOG(INFO) << opts.ToString();

  auto wave = ReadWave(po.GetArg(1), opts.sample_rate);
  std::cerr << wave.sizes() << "\n";
  std::cerr << wave.scalar_type() << "\n";

  return 0;
}

/* Read wave samples from a file. Samples are divided by 32768 so that
 * they are normalized to the range [-1, 1]. All models from icefall
 * use normalized samples while extracting features..
 */
static torch::Tensor ReadWave(const std::string &filename,
                              float expected_sample_rate) {
  bool binary = true;
  kaldiio::Input ki(filename, &binary);
  kaldiio::WaveHolder wh;
  if (!wh.Read(ki.Stream())) {
    SHERPA_LOG(FATAL) << "Failed to read " << filename;
  }

  auto &wave_data = wh.Value();
  if (wave_data.SampFreq() != expected_sample_rate) {
    SHERPA_LOG(FATAL) << filename << "is expect to have sample rate "
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

  return tensor / 32768;
}
