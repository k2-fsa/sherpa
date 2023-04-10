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

  // For Wav2Vec 2.0, we set it to true so that it returns audio samples
  // directly.
  //
  // The user does not need to set it. We set it internally when we
  // load a Wav2Vec 2.0 model.
  bool return_waveform = false;

  // For models from NeMo
  // Possible values:
  // - per_feature
  // - all_features (not implemented yet)
  // - fixed_mean (not implemented)
  // - fixed_std (not implemented)
  // - or just leave it to empty
  // See
  // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L59
  // for details
  std::string nemo_normalize;

  void Register(ParseOptions *po);

  /** A string representation for debugging purpose. */
  std::string ToString() const;
};

std::ostream &operator<<(std::ostream &os, const FeatureConfig &config);

}  // namespace sherpa

#endif  // SHERPA_CPP_API_FEATURE_CONFIG_H_
