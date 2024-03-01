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

  po->Register(
      "snip-edges", &opts->snip_edges,
      "If true, end effects will be handled by outputting only frames that "
      "completely fit in the file, and the number of frames depends on the "
      "frame-length.  If false, the number of frames depends only on the "
      "frame-shift, and we reflect the data at the ends.");
}

static void RegisterMelBanksOptions(ParseOptions *po,
                                    kaldifeat::MelBanksOptions *opts) {
  po->Register("num-mel-bins", &opts->num_bins,
               "Number of triangular mel-frequency bins");
  po->Register(
      "high-freq", &opts->high_freq,
      "High cutoff frequency for mel bins (if <= 0, offset from Nyquist)");
}

void FeatureConfig::Register(ParseOptions *po) {
  fbank_opts.frame_opts.dither = 0;
  RegisterFrameExtractionOptions(po, &fbank_opts.frame_opts);

  fbank_opts.mel_opts.num_bins = 80;
  RegisterMelBanksOptions(po, &fbank_opts.mel_opts);

  fbank_opts.mel_opts.high_freq = -400;
  fbank_opts.frame_opts.remove_dc_offset = true;
  fbank_opts.frame_opts.round_to_power_of_two = true;
  fbank_opts.energy_floor = 1e-10;
  fbank_opts.frame_opts.snip_edges = false;
  fbank_opts.frame_opts.samp_freq = 16000;
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
