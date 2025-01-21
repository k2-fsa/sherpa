// sherpa/csrc/sherpa-vad.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <iostream>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/fbank-features.h"
#include "sherpa/csrc/voice-activity-detector.h"

int32_t main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
This program uses a VAD models to add timestamps to a audio file
Usage:

sherpa-vad \
  --silero-vad-model=/path/to/model.pt \
  --use-gpu=false \
  ./foo.wav

)usage";

  sherpa::ParseOptions po(kUsageMessage);
  sherpa::VoiceActivityDetectorConfig config;
  config.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() != 1) {
    std::cerr << "Please provide only 1 test wave\n";
    exit(-1);
  }

  std::cerr << config.ToString() << "\n";
  config.Validate();

  sherpa::VoiceActivityDetector vad(config);

  torch::Tensor samples = sherpa::ReadWave(po.GetArg(1), 16000).first;

  vad.Process(samples);
}
