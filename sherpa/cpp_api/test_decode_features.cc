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
#include "sherpa/cpp_api/offline_recognizer.h"
#include "sherpa/csrc/fbank_features.h"
#include "torch/script.h"

/** Read wave samples from a file.
 *
 * If the file has multiple channels, only the first channel is returned.
 * Samples are normalized to the range [-1, 1).
 *
 * @param filename Path to the wave file. Only "*.wav" format is supported.
 * @param expected_sample_rate  Expected sample rate of the wave file. It aborts
 *                              if the sample rate of the given file is not
 *                              equal to this value.
 *
 * @return Return a 1-D torch.float32 tensor containing audio samples
 * in the range [-1, 1)
 */
static torch::Tensor ReadWave(const std::string &filename,
                              float expected_sample_rate) {
  bool binary = true;
  kaldiio::Input ki(filename, &binary);
  kaldiio::WaveHolder wh;
  if (!wh.Read(ki.Stream())) {
    std::cerr << "Failed to read " << filename;
    exit(EXIT_FAILURE);
  }

  auto &wave_data = wh.Value();
  if (wave_data.SampFreq() != expected_sample_rate) {
    std::cerr << filename << "is expected to have sample rate "
              << expected_sample_rate << ". Given " << wave_data.SampFreq();
    exit(EXIT_FAILURE);
  }

  auto &d = wave_data.Data();

  if (d.NumRows() > 1) {
    std::cerr << "Only the first channel from " << filename << " is used";
  }

  auto tensor = torch::from_blob(const_cast<float *>(d.RowData(0)),
                                 {d.NumCols()}, torch::kFloat);

  return tensor / 32768;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: ./bin/test_decode_file /path/to/nn_model "
                 "/path/to/tokens.txt foo.wav [bar.wav [foobar.wav] ... ]\n";
    exit(EXIT_FAILURE);
  }
  std::string nn_model = argv[1];
  std::string tokens = argv[2];
  float sample_rate = 16000;
  bool use_gpu = false;

  sherpa::DecodingOptions opts;
  opts.method = sherpa::kGreedySearch;
  sherpa::OfflineRecognizer recognizer(nn_model, tokens, opts, use_gpu,
                                       sample_rate);

  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.samp_freq = sample_rate;
  fbank_opts.mel_opts.num_bins = 80;

  kaldifeat::Fbank fbank(fbank_opts);  // always on CPU

  if (argc == 4) {
    std::cout << "Decode single file\n";

    auto samples = ReadWave(argv[3], sample_rate);
    auto feature = fbank.ComputeFeatures(samples, 1.0);

    auto result = recognizer.DecodeFeatures(feature.data_ptr<float>(),
                                            feature.size(0), 80);
    std::cout << argv[3] << "\n" << result.text << "\n";
    return 0;
  }

  std::cout << "Decode multiple files\n";

  std::vector<torch::Tensor> features;
  std::vector<int32_t> features_length;
  for (int i = 3; i != argc; ++i) {
    auto samples = ReadWave(argv[i], sample_rate);
    auto feature = fbank.ComputeFeatures(samples, 1.0);
    features.push_back(feature);

    features_length.push_back(feature.size(0));
  }
  torch::Tensor padded_features = torch::nn::utils::rnn::pad_sequence(
      features, /*batch_first*/ true,
      /*padding_value*/ -23.025850929940457f);

  auto results = recognizer.DecodeFeaturesBatch(
      padded_features.data_ptr<float>(), features_length.data(),
      padded_features.size(0), padded_features.size(1), 80);

  for (size_t i = 0; i != features_length.size(); ++i) {
    std::cout << argv[i + 3] << "\n" << results[i].text << "\n\n";
  }

  return 0;
}
