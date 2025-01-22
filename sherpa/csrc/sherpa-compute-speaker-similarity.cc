// sherpa/csrc/sherpa-compute-speaker-similarity.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <chrono>  // NOLINT
#include <iostream>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/fbank-features.h"
#include "sherpa/csrc/speaker-embedding-extractor.h"

int32_t main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
This program uses a speaker embedding model to compute
similarity between two wave files.

sherpa-compute-speaker-similarity \
  --model=/path/to/model.pt \
  ./foo.wav \
  ./bar.wav \
)usage";

  int32_t num_threads = 1;
  sherpa::ParseOptions po(kUsageMessage);
  sherpa::SpeakerEmbeddingExtractorConfig config;
  config.Register(&po);
  po.Register("num-threads", &num_threads, "Number of threads for PyTorch");
  po.Read(argc, argv);

  if (po.NumArgs() != 2) {
    std::cerr << "Please provide only 2 test waves\n";
    exit(-1);
  }

  std::cerr << config.ToString() << "\n";
  if (!config.Validate()) {
    std::cerr << "Please check your config\n";
    return -1;
  }

  int32_t sr = 16000;
  sherpa::SpeakerEmbeddingExtractor extractor(config);

  torch::Tensor samples1 = sherpa::ReadWave(po.GetArg(1), sr).first;

  auto stream1 = extractor.CreateStream();
  stream1->AcceptSamples(samples1.data_ptr<float>(), samples1.numel());

  torch::Tensor samples2 = sherpa::ReadWave(po.GetArg(2), sr).first;

  auto stream2 = extractor.CreateStream();
  stream2->AcceptSamples(samples2.data_ptr<float>(), samples2.numel());

  torch::Tensor embedding1;
  torch::Tensor embedding2;
  if (false) {
    embedding1 = extractor.Compute(stream1.get()).squeeze(0);
    embedding2 = extractor.Compute(stream2.get()).squeeze(0);
  } else {
    std::vector<sherpa::OfflineStream *> ss{stream1.get(), stream2.get()};
    auto embeddings = extractor.Compute(ss.data(), ss.size());

    embedding1 = embeddings.index({0});
    embedding2 = embeddings.index({1});
    std::cout << "embedding1.shape " << embedding1.sizes() << "\n";
    std::cout << "embedding2.shape " << embedding2.sizes() << "\n";
  }

  auto score =
      torch::nn::functional::cosine_similarity(
          embedding1, embedding2,
          torch::nn::functional::CosineSimilarityFuncOptions{}.dim(0).eps(1e-6))
          .item()
          .toFloat();
  std::cout << "score: " << score << "\n";

  return 0;
}
