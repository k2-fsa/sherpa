/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sherpa/csrc/offline_asr.h"

#include <utility>

#include "sherpa/csrc/fbank_features.h"
#include "sherpa/csrc/file_utils.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/rnnt_beam_search.h"

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

void OfflineAsrOptions::Register(ParseOptions *po) {
  po->Register("nn-model", &nn_model, "Path to the torchscript model");

  po->Register("tokens", &tokens, "Path to tokens.txt.");

  po->Register("decoding-method", &decoding_method,
               "Decoding method to use. Possible values are: greedy_search, "
               "modified_beam_search");

  po->Register("num-active-paths", &num_active_paths,
               "Number of active paths for modified_beam_search. "
               "Used only when --decoding-method is modified_beam_search");

  po->Register("use-gpu", &use_gpu,
               "true to use GPU for computation. false to use CPU.\n"
               "If true, it uses the first device. You can use the environment "
               "variable CUDA_VISIBLE_DEVICES to select which device to use.");

  fbank_opts.frame_opts.dither = 0;
  RegisterFrameExtractionOptions(po, &fbank_opts.frame_opts);

  fbank_opts.mel_opts.num_bins = 80;
  RegisterMelBanksOptions(po, &fbank_opts.mel_opts);
}

void OfflineAsrOptions::Validate() const {
  if (nn_model.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --nn-model";
  }

  if (!FileExists(nn_model)) {
    SHERPA_LOG(FATAL) << "\n--nn-model=" << nn_model << "\n"
                      << nn_model << " does not exist!";
  }

  if (tokens.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --tokens";
  }

  if (!FileExists(tokens)) {
    SHERPA_LOG(FATAL) << "\n--tokens=" << tokens << "\n"
                      << tokens << " does not exist!";
  }

  if (decoding_method != "greedy_search" &&
      decoding_method != "modified_beam_search") {
    SHERPA_LOG(FATAL)
        << "Unsupported decoding method: " << decoding_method
        << ". Supported values are: greedy_search, modified_beam_search";
  }

  if (decoding_method == "modified_beam_search") {
    SHERPA_CHECK_GT(num_active_paths, 0);
  }
}

std::string OfflineAsrOptions::ToString() const {
  std::ostringstream os;
  os << "--nn-model=" << nn_model << "\n";
  os << "--tokens=" << tokens << "\n";

  os << "--decoding-method=" << decoding_method << "\n";

  if (decoding_method == "modified_beam_search") {
    os << "--num-active-paths=" << num_active_paths << "\n";
  }

  os << "--use-gpu=" << std::boolalpha << use_gpu << "\n";

  return os.str();
}

OfflineAsr::OfflineAsr(const OfflineAsrOptions &opts)
    : opts_(opts),
      model_(opts.nn_model,
             opts.use_gpu ? torch::Device("cuda:0") : torch::Device("cpu")),
      sym_(opts.tokens),
      fbank_(opts.fbank_opts) {}

std::vector<OfflineAsrResult> OfflineAsr::DecodeWaves(
    const std::vector<std::string> &filenames, float expected_sample_rate) {
  std::vector<torch::Tensor> waves;
  for (const auto &f : filenames) {
    waves.push_back(ReadWave(f, expected_sample_rate).first);
  }

  return DecodeWaves(waves);
}

std::vector<OfflineAsrResult> OfflineAsr::DecodeWaves(
    const std::vector<torch::Tensor> &waves) {
  std::vector<torch::Tensor> features = ComputeFeatures(fbank_, waves);
  return DecodeFeatures(features);
}

std::vector<OfflineAsrResult> OfflineAsr::DecodeFeatures(
    const std::vector<torch::Tensor> &features) {
  torch::Tensor padded_features = torch::nn::utils::rnn::pad_sequence(
      features, /*batch_first*/ true,
      /*padding_value*/ -23.025850929940457f);

  std::vector<int64_t> feature_length_vec(features.size());
  for (size_t i = 0; i != features.size(); ++i) {
    feature_length_vec[i] = features[i].size(0);
  }

  torch::Tensor feature_lengths = torch::tensor(feature_length_vec);

  return DecodeFeatures(padded_features, feature_lengths);
}

std::vector<OfflineAsrResult> OfflineAsr::DecodeFeatures(
    torch::Tensor features, torch::Tensor features_length) {
  auto device = model_.Device();
  features = features.to(device);
  features_length = features_length.to(device).to(torch::kLong);

  torch::Tensor encoder_out;
  torch::Tensor encoder_out_length;

  std::tie(encoder_out, encoder_out_length) =
      model_.ForwardEncoder(features, features_length);
  encoder_out_length = encoder_out_length.cpu();

  std::vector<std::vector<int32_t>> token_ids;

  if (opts_.decoding_method == "greedy_search") {
    token_ids = GreedySearch(model_, encoder_out, encoder_out_length);
  } else if (opts_.decoding_method == "modified_beam_search") {
    token_ids = ModifiedBeamSearch(model_, encoder_out, encoder_out_length,
                                   opts_.num_active_paths);
  } else {
    SHERPA_LOG(FATAL) << "Unsupported decoding method: "
                      << opts_.decoding_method;
  }

  int32_t batch_size = features.size(0);
  std::vector<OfflineAsrResult> results(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    auto &text = results[i].text;
    for (auto t : token_ids[i]) {
      text += sym_[t];
    }
  }

  return results;
}

}  // namespace sherpa
