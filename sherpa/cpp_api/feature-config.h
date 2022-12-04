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

  // In sherpa, we always assume the input audio samples are normalized to
  // the range [-1, 1].
  // ``normalize_samples`` determines how we transform the input samples
  // inside sherpa.
  // If true, we don't do anything to the input audio samples and use them
  // as they are.
  //
  // If false, we scale the input samples by 32767 inside sherpa
  bool normalize_samples = true;

  void Register(ParseOptions *po);

  /** A string representation for debugging purpose. */
  std::string ToString() const;
};

std::ostream &operator<<(std::ostream &os, const FeatureConfig &config);

}  // namespace sherpa

#endif  // SHERPA_CPP_API_FEATURE_CONFIG_H_
