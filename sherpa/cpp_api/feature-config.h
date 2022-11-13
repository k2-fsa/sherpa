// sherpa/cpp_api/feature-config.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef SHERPA_CPP_API_FEATURE_CONFIG_H_
#define SHERPA_CPP_API_FEATURE_CONFIG_H_

#include <string>

#include "kaldifeat/csrc/feature-fbank.h"
#include "sherpa/cpp_api/parse-options.h"

namespace sherpa {

struct FeatureConfig {
  kaldifeat::FbankOptions fbank_opts;

  // true to use samples in the range [-1, 1]
  // false to use samples in the range [-32768, 32767]
  bool normalize_samples = true;

  void Register(ParseOptions *po);

  /** A string representation for debugging purpose. */
  std::string ToString() const;
};

std::ostream &operator<<(std::ostream &os, const FeatureConfig &config);

}  // namespace sherpa

#endif  // SHERPA_CPP_API_FEATURE_CONFIG_H_
