// sherpa/cpp_api/bin/offline-recognizer.h
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/cpp_api/offline-recognizer.h"

#include <iostream>

#include "sherpa/cpp_api/parse-options.h"
#include "torch/all.h"

static constexpr const char *kUsageMessage = R"(
Decode wave file(s) using offline recognizer from sherpa.

Usage:

./bin/sherpa-offline-recongizer --help

./bin/sherpa-offline-recognizer \
  --use-gpu=false \
  --nn-model=/path/to/cpu.jit \
  --tokens=/path/to/tokens.txt \
  --decoding-method=greedy_search \
  /path/to/foo.wav \
  /path/to/bar.wav \
  /path/to/foobar.wav
)";

static std::ostream &operator<<(std::ostream &os,
                                const std::vector<int32_t> &v) {
  std::string sep = "";
  os << "[";
  for (auto i : v) {
    os << sep << i;
    sep = " ";
  }
  os << "]";

  return os;
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  sherpa::ParseOptions po(kUsageMessage);
  sherpa::OfflineRecognizerConfig config;
  config.Register(&po);

  po.Read(argc, argv);

  if (po.NumArgs() == 0) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  config.Validate();

  sherpa::OfflineRecognizer recognizer(config);

  std::vector<std::unique_ptr<sherpa::OfflineStream>> ss;
  std::vector<sherpa::OfflineStream *> p;
  for (int i = 1; i <= po.NumArgs(); ++i) {
    ss.push_back(recognizer.CreateStream());
    p.push_back(ss.back().get());
    ss.back()->AcceptWaveFile(po.GetArg(i));
  }

  recognizer.DecodeStreams(p.data(), po.NumArgs());

  std::ostringstream os;

  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    const auto &r = ss[i - 1]->GetResult();
    os << "filename: " << po.GetArg(1) << "\n";
    os << "text: " << r.text << "\n";
    os << "token IDs: " << r.tokens << "\n";
    os << "timestamps (after subsampling): " << r.timestamps << "\n\n";
  }
  std::cout << os.str() << "\n";
  return 0;
}
