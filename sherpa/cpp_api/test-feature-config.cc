// sherpa/cpp_api/test-feature-config.h
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/parse-options.h"

int main(int argc, char *argv[]) {
  sherpa::ParseOptions po("");
  sherpa::FeatureConfig feat_config;
  feat_config.Register(&po);
  po.Read(argc, argv);
  po.PrintUsage();

  std::cout << feat_config << "\n";

  return 0;
}
