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

#include "kaldifeat/csrc/feature-fbank.h"
#include "sherpa/csrc/file_utils.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/parse_options.h"

struct Options {
  /// Path to torchscript model
  std::string nn_model;

  /// Path to BPE model file. Used only for BPE based models
  std::string bpe_model;

  void Register(sherpa::ParseOptions *po) {
    po->Register("nn-model", &nn_model, "Path to the torchscript model");
    po->Register("bpe-model", &bpe_model,
                 "Path to the BPE model. Used only for BPE based models");
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
    return os.str();
  }
};

int main(int argc, char *argv[]) {
  const char *usage = R"(Automatic speech recognition with sherpa.
Usage:
  ./bin/sherpa --help
  )";
  sherpa::ParseOptions po(usage);

  Options opts;
  opts.Register(&po);

  po.Read(argc, argv);
  opts.Validate();

  SHERPA_LOG(INFO) << "\n" << opts.ToString();

  return 0;
}
