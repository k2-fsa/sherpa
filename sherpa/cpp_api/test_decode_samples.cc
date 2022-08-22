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

  if (argc == 4) {
    std::cout << "Decode single file\n";
    auto tensor = ReadWave(argv[3], sample_rate);
    auto result =
        recognizer.DecodeSamples(tensor.data_ptr<float>(), tensor.size(0));
    std::cout << argv[3] << "\n" << result.text << "\n";
    return 0;
  }

  std::cout << "Decode multiple files\n";

  std::vector<torch::Tensor> tensors;
  std::vector<const float *> tensors_addr;
  std::vector<int32_t> tensors_length;
  for (int i = 3; i != argc; ++i) {
    tensors.push_back(ReadWave(argv[i], sample_rate));
    tensors_addr.push_back(tensors.back().data_ptr<float>());
    tensors_length.push_back(tensors.back().size(0));
  }

  auto results = recognizer.DecodeSamplesBatch(
      tensors_addr.data(), tensors_length.data(), tensors_length.size());

  for (size_t i = 0; i != tensors_length.size(); ++i) {
    std::cout << argv[i + 3] << "\n" << results[i].text << "\n\n";
  }
  return 0;
}
