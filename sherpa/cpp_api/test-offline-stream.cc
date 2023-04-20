// sherpa/cpp_api/test-offline-stream.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-stream.h"

int main(int argc, char *argv[]) {
  sherpa::FeatureConfig feat_config;
  kaldifeat::Fbank fbank(feat_config.fbank_opts);
  sherpa::OfflineRecognitionResult r;
  r.text = "hello world";

  if (argc == 2) {
    std::cout << "===test from wave file===\n";
    sherpa::OfflineStream s(&fbank, feat_config);
    s.AcceptWaveFile(argv[1]);
    auto f = s.GetFeatures();
    std::cout << "f.sizes(): " << f.sizes() << "\n";
    s.SetResult(r);
    std::cout << s.GetResult().text << "\n";
  }

  {
    std::cout << "===test from samples===\n";
    torch::Tensor samples = torch::rand({160000}, torch::kFloat);
    sherpa::OfflineStream s(&fbank, feat_config);
    s.AcceptSamples(samples.data_ptr<float>(), samples.numel());
    auto f = s.GetFeatures();
    std::cout << "f.sizes(): " << f.sizes() << "\n";
    s.SetResult(r);
    std::cout << s.GetResult().text << "\n";
  }

  {
    std::cout << "===test from features===\n";
    torch::Tensor features = torch::rand(
        {50, feat_config.fbank_opts.mel_opts.num_bins}, torch::kFloat);
    sherpa::OfflineStream s(&fbank, feat_config);
    s.AcceptFeatures(features.data_ptr<float>(), features.size(0),
                     features.size(1));
    auto f = s.GetFeatures();
    std::cout << "f.sizes(): " << f.sizes() << "\n";
    s.SetResult(r);
    std::cout << s.GetResult().text << "\n";
  }

  return 0;
}
