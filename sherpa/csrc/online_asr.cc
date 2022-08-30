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

#include "sherpa/csrc/online_asr.h"

#include "sherpa/csrc/file_utils.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/parse_options.h"
#include "sherpa/csrc/rnnt_beam_search.h"
#include "sherpa/csrc/rnnt_conv_emformer_model.h"

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

void OnlineAsrOptions::Register(ParseOptions *po) {
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

void OnlineAsrOptions::Validate() const {
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

std::string OnlineAsrOptions::ToString() const {
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

OnlineAsr::OnlineAsr(const OnlineAsrOptions &opts)
    : opts_(opts),
      model_(std::make_unique<RnntConvEmformerModel>(
          opts.nn_model,
          opts.use_gpu ? torch::Device("cuda:0") : torch::Device("cpu"))),
      sym_(opts.tokens) {}

std::unique_ptr<OnlineStream> OnlineAsr::CreateStream() {
  auto s = std::make_unique<OnlineStream>(
      opts_.fbank_opts.frame_opts.samp_freq, opts_.fbank_opts.mel_opts.num_bins,
      opts_.fbank_opts.frame_opts.max_feature_vectors);

  auto device = model_->Device();
  auto &hyps = s->GetHyps();
  auto &decoder_out = s->GetDecoderOut();

  int32_t blank_id = model_->BlankId();
  int32_t context_size = model_->ContextSize();
  hyps.resize(context_size, blank_id);

  torch::Tensor decoder_input =
      torch::tensor(hyps, torch::kLong).unsqueeze(0).to(device);
  torch::Tensor initial_decoder_out = model_->ForwardDecoder(decoder_input);
  decoder_out = model_->ForwardDecoderProj(initial_decoder_out.squeeze(1));

  auto state = model_->GetEncoderInitStates2();
  s->SetState(state);

  return s;
}

bool OnlineAsr::IsReady(OnlineStream *s) {
  int32_t chunk_length = model_->ChunkLength();  // e.g., 32
  int32_t pad_length = model_->PadLength();      // e.g., 19
  int32_t chunk_length_pad = chunk_length + pad_length;
  return s->NumFramesReady() - s->GetNumProcessedFrames() >= chunk_length_pad;
}

void OnlineAsr::DecodeStream(OnlineStream *s) {
  SHERPA_CHECK_EQ(opts_.decoding_method, "greedy_search");
  auto device = model_->Device();
  int32_t chunk_length = model_->ChunkLength();  // 32
  int32_t pad_length = model_->PadLength();      // 19
                                                 //
  int32_t chunk_length_pad = chunk_length + pad_length;
  int32_t &num_processed_frames = s->GetNumProcessedFrames();

  SHERPA_CHECK(IsReady(s));

  std::vector<torch::Tensor> features_vec(chunk_length_pad);
  for (int32_t i = 0; i != chunk_length_pad; ++i) {
    features_vec[i] = s->GetFrame(num_processed_frames + i);
  }

  torch::Tensor features = torch::cat(features_vec, /*dim*/ 0);

  features = features.unsqueeze(0);  // batch_size == 1 for now
  features = features.to(device);
  torch::Tensor features_length =
      torch::tensor({chunk_length_pad}, torch::kLong).to(device);

  torch::IValue state = s->GetState();
  torch::IValue stacked_states = s->StackStates({state});
  torch::Tensor processed_frames =
      torch::tensor({num_processed_frames}, torch::kLong).to(device);

  torch::Tensor encoder_out;
  torch::Tensor encoder_out_lens;
  torch::IValue next_states;

  std::tie(encoder_out, encoder_out_lens, next_states) =
      model_->StreamingForwardEncoder2(features, features_length,
                                       processed_frames, stacked_states);

  num_processed_frames += chunk_length;

  s->SetState(s->UnStackStates(next_states)[0]);

  std::vector<std::vector<int32_t>> hyps_list = {s->GetHyps()};
  torch::Tensor &decoder_out = s->GetDecoderOut();
  decoder_out =
      StreamingGreedySearch(*model_, encoder_out, decoder_out, &hyps_list);
  s->GetHyps() = hyps_list[0];
}

std::string OnlineAsr::GetResults(OnlineStream *s) const {
  int32_t context_size = model_->ContextSize();

  auto &hyps = s->GetHyps();
  std::string text;

  for (int32_t i = 0; i != hyps.size(); ++i) {
    if (i < context_size) continue;
    text += sym_[hyps[i]];
  }
  return text;
}

}  // namespace sherpa
