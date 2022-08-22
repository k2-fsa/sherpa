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

#include "sherpa/cpp_api/offline_recognizer.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: ./bin/test_decode_files /path/to/nn_model "
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
    auto result = recognizer.DecodeFile(argv[3]);
    std::cout << argv[3] << "\n" << result.text << "\n";
    return 0;
  }

  std::cout << "Decode multiple files\n";

  std::vector<std::string> filenames;
  for (int i = 3; i != argc; ++i) {
    filenames.push_back(argv[i]);
  }

  auto results = recognizer.DecodeFileBatch(filenames);
  for (size_t i = 0; i != filenames.size(); ++i) {
    std::cout << filenames[i] << "\n" << results[i].text << "\n\n";
  }
  return 0;
}
