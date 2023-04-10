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

  po->Register("normalize-samples", &normalize_samples,
               "true to use samples in the range [-1, 1]. "
               "false to use samples in the range [-32768, 32767]. "
               "Note: kaldi uses un-normalized samples.");

  po->Register(
      "nemo-normalize", &nemo_normalize,
      "See "
      "https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/"
      "preprocessing/features.py#L59"
      "Current supported value: per_feature or leave it to empty (unset)");
}

std::string FeatureConfig::ToString() const {
  std::ostringstream os;
  os << "FeatureConfig(";
  os << "fbank_opts=" << fbank_opts.ToString() << ", ";
  os << "normalize_samples=" << (normalize_samples ? "True" : "False") << ", ";
  os << "nemo_normalize=\"" << nemo_normalize << "\")";
  return os.str();
}

std::ostream &operator<<(std::ostream &os, const FeatureConfig &config) {
  os << config.ToString();
  return os;
}

}  // namespace sherpa
