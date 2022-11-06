// sherpa/cpp_api/feature-config.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/feature-config.h"

#include <string>

namespace sherpa {

static void RegisterFrameExtractionOptions(
    ParseOptions *po, kaldifeat::FrameExtractionOptions *opts) {
  po->Register("sample-frequency", &opts->samp_freq,
               "Waveform data sample frequency (must match the waveform file, "
               "if specified there)");

  po->Register("frame-length", &opts->frame_length_ms,
               "Frame length in milliseconds");

  po->Register("frame-shift", &opts->frame_shift_ms,
               "Frame shift in milliseconds");

  po->Register(
      "dither", &opts->dither,
      "Dithering constant (0.0 means no dither). "
      "Caution: Samples are normalized to the range [-1, 1). "
      "Please select a small value for dither if you want to enable it");
}

static void RegisterMelBanksOptions(ParseOptions *po,
                                    kaldifeat::MelBanksOptions *opts) {
  po->Register("num-mel-bins", &opts->num_bins,
               "Number of triangular mel-frequency bins");
}

void FeatureConfig::Register(ParseOptions *po) {
  fbank_opts.frame_opts.dither = 0;
  RegisterFrameExtractionOptions(po, &fbank_opts.frame_opts);

  fbank_opts.mel_opts.num_bins = 80;
  RegisterMelBanksOptions(po, &fbank_opts.mel_opts);
}

std::string FeatureConfig::ToString() const { return fbank_opts.ToString(); }

std::ostream &operator<<(std::ostream &os, const FeatureConfig &config) {
  os << config.ToString();
  return os;
}

}  // namespace sherpa
