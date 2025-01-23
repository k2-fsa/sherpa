// sherpa/csrc/sherpa-vad.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <chrono>  // NOLINT
#include <iostream>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/fbank-features.h"
#include "sherpa/csrc/offline-speaker-diarization.h"
#include "torch/torch.h"
int32_t main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
This program uses a speaker segmentation model and a speaker embedding extractor
model for speaker diarization.
Usage:

sherpa-speaker-diarization \
  --vad-use-gpu=false \
  --num-threads=1 \
  --embedding.model=./3d_speaker-speech_eres2netv2_sv_zh-cn_16k-common.pt \
  --segmentation.pyannote-model=./sherpa-pyannote-segmentation-3-0/model.pt \
  ./foo.wav

)usage";
  int32_t num_threads = 1;
  sherpa::ParseOptions po(kUsageMessage);
  sherpa::OfflineSpeakerDiarizationConfig config;
  config.Register(&po);
  po.Register("num-threads", &num_threads, "Number of threads for PyTorch");
  po.Read(argc, argv);

  if (po.NumArgs() != 1) {
    std::cerr << "Please provide only 1 test wave\n";
    exit(-1);
  }

  std::cerr << config.ToString() << "\n";
  if (!config.Validate()) {
    std::cerr << "Please check your config\n";
    return -1;
  }

  sherpa::OfflineSpeakerDiarization sd(config);

  int32_t sr = 16000;
  torch::Tensor samples = sherpa::ReadWave(po.GetArg(1), sr).first;
  sd.Process(samples.unsqueeze(0));
  std::cout << "here\n";

  return 0;
}
