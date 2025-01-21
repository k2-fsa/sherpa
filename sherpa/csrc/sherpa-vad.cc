// sherpa/csrc/sherpa-vad.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <chrono>  // NOLINT
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

  const auto begin = std::chrono::steady_clock::now();

  auto segments = vad.Process(samples);

  const auto end = std::chrono::steady_clock::now();

  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = samples.size(0) / 16000.0f;

  const float rtf = elapsed_seconds / duration;
  for (const auto &s : segments) {
    fprintf(stderr, "%.3f -- %.3f\n", s.start, s.end);
  }

  fprintf(stderr, "Elapsed seconds: %.3f\n", elapsed_seconds);
  fprintf(stderr, "Audio duration: %.3f s\n", duration);
  fprintf(stderr, "Real time factor (RTF): %.3f/%.3f = %.3f\n", elapsed_seconds,
          duration, rtf);
}
