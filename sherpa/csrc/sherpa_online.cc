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

#include <string>

#include "sherpa/csrc/fbank_features.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_stream.h"
#include "sherpa/csrc/rnnt_beam_search.h"
#include "sherpa/csrc/rnnt_conv_emformer_model.h"
#include "sherpa/csrc/symbol_table.h"
#include "torch/script.h"

std::ostream &operator<<(std::ostream &os, const std::vector<int32_t> &v) {
  os << "[";
  std::string sep = "";
  for (auto i : v) {
    os << sep << i;
    sep = ", ";
  }
  os << "]";
  return os;
}

static void CheckStates(torch::IValue a, torch::IValue b) {
  TORCH_CHECK(a.tagKind() == b.tagKind(), a.tagKind(), " vs ", b.tagKind());

  auto a_tuple_ptr = a.toTuple();
  torch::List<torch::IValue> a_list_attn = a_tuple_ptr->elements()[0].toList();
  torch::List<torch::IValue> a_list_conv = a_tuple_ptr->elements()[1].toList();

  auto b_tuple_ptr = b.toTuple();
  torch::List<torch::IValue> b_list_attn = b_tuple_ptr->elements()[0].toList();
  torch::List<torch::IValue> b_list_conv = b_tuple_ptr->elements()[1].toList();

  int32_t num_layers = a_list_attn.size();
  TORCH_CHECK(num_layers == b_list_attn.size());
  for (int32_t i = 0; i != num_layers; ++i) {
    torch::IValue a_i = a_list_attn[i];
    torch::IValue b_i = b_list_attn[i];

    // for attn
    auto a_i_v = c10::impl::toTypedList<torch::Tensor>(a_i.toList()).vec();
    auto b_i_v = c10::impl::toTypedList<torch::Tensor>(b_i.toList()).vec();
    for (size_t k = 0; k != a_i_v.size(); ++k) {
      TORCH_CHECK(torch::equal(a_i_v[k], b_i_v[k]));
    }

    // for conv
    auto a_t = static_cast<const torch::IValue &>(a_list_conv[i]).toTensor();
    auto b_t = static_cast<const torch::IValue &>(b_list_conv[i]).toTensor();

    TORCH_CHECK(torch::equal(a_t, b_t));
  }  // for (int32_t i = 0; i != num_layers; ++i)
}

static void TestStackUnstackStates(sherpa::RnntConvEmformerModel &model,
                                   sherpa::OnlineStream &s) {
  auto device = model.Device();

  // Tuple[List[List[torch.Tensor]], List[torch.Tensor]]
  torch::IValue states0 = model.GetEncoderInitStates2();
  torch::IValue states1 = model.GetEncoderInitStates2();
  torch::IValue states2 = model.GetEncoderInitStates2();

  torch::IValue stacked_states = s.StackStates({states0, states1, states2});
  std::vector<torch::IValue> states = s.UnStackStates(stacked_states);
  TORCH_CHECK(states.size() == 3);

  CheckStates(states0, states[0]);
  CheckStates(states1, states[1]);
  CheckStates(states2, states[2]);
}

static void InitStream(sherpa::RnntConvEmformerModel &model,
                       sherpa::OnlineStream &s) {
  auto device = model.Device();
  auto &hyps = s.GetHyps();
  auto &decoder_out = s.GetDecoderOut();

  int32_t blank_id = model.BlankId();
  int32_t context_size = model.ContextSize();
  hyps.resize(context_size, blank_id);

  torch::Tensor decoder_input =
      torch::tensor(hyps, torch::kLong).unsqueeze(0).to(device);
  torch::Tensor initial_decoder_out = model.ForwardDecoder(decoder_input);
  decoder_out = model.ForwardDecoderProj(initial_decoder_out.squeeze(1));
  std::cerr << "decoder_out.sizes(): " << s.GetDecoderOut().sizes() << "\n";
  std::cerr << "hyps: " << s.GetHyps() << "\n";

  auto state = model.GetEncoderInitStates2();
  s.SetState(state);
}

static void DecodeStream(sherpa::RnntConvEmformerModel &model,
                         sherpa::OnlineStream &s) {
  auto device = model.Device();
  // implement greedy search first
  int32_t chunk_length = model.ChunkLength();                 // 32
  int32_t right_context_length = model.RightContextLength();  // 8
  int32_t pad_length = model.PadLength();                     // 19
                                                              //
  int32_t chunk_length_pad = chunk_length + pad_length;
  int32_t &num_processed_frames = s.GetNumProcessedFrames();

  SHERPA_CHECK_GE(s.NumFramesReady() - num_processed_frames, chunk_length_pad);

  std::vector<torch::Tensor> features_vec(chunk_length_pad);
  for (int32_t i = 0; i != chunk_length_pad; ++i) {
    features_vec[i] = s.GetFrame(num_processed_frames + i);
  }

  torch::Tensor features = torch::cat(features_vec, /*dim*/ 0);
  std::cout << "features.sizes(): " << features.sizes() << "\n";

  features = features.unsqueeze(0);  // batch_size == 1 for now
  features = features.to(device);
  torch::Tensor features_length =
      torch::tensor({chunk_length_pad}, torch::kLong).to(device);

  torch::IValue state = s.GetState();
  torch::IValue stacked_states = s.StackStates({state});
  torch::Tensor processed_frames =
      torch::tensor({num_processed_frames}, torch::kLong).to(device);

  torch::Tensor encoder_out;
  torch::Tensor encoder_out_lens;
  torch::IValue next_states;

  std::cout << "features.sizes(): " << features.sizes() << "\n";
  std::cout << "features_length.sizes(): " << features_length.sizes() << "\n";
  std::cout << "processed_frames.sizes(): " << processed_frames.sizes() << "\n";
  std::tie(encoder_out, encoder_out_lens, next_states) =
      model.StreamingForwardEncoder2(features, features_length,
                                     processed_frames, stacked_states);
  std::cout << "encoder_out_lens: " << encoder_out_lens << "\n";

  num_processed_frames += chunk_length;

  s.SetState(s.UnStackStates(next_states)[0]);

  std::vector<std::vector<int32_t>> hyps_list = {s.GetHyps()};
  torch::Tensor &decoder_out = s.GetDecoderOut();
  // decoder_out = decoder_out.unsqueeze(1);
  std::cout << "encoder_out.sizes(): " << encoder_out.sizes() << "\n";
  std::cout << "decoder_out.sizes(): " << decoder_out.sizes() << "\n";
  decoder_out =
      StreamingGreedySearch(model, encoder_out, decoder_out, &hyps_list);
  // decoder_out = decoder_out.squeeze(1);
  s.GetHyps() = hyps_list[0];
  std::cout << "hyps: " << hyps_list[0] << "\n";
}

int main() {
  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  float sampling_rate = 16000;
  int32_t feature_dim = 80;
  int32_t max_feature_vectors = 100;  // TODO(fangjun): tune it

  torch::Device device(torch::kCPU);

  sherpa::OnlineStream s(sampling_rate, feature_dim, max_feature_vectors);

  std::string nn_model = "./cpu-conv-emformer-jit.pt";
  std::string tokens = "./tokens.txt";
  std::string wave_filename = "./1089-134686-0001.wav";

  sherpa::SymbolTable sym(tokens);

  torch::Tensor wave = sherpa::ReadWave(wave_filename, sampling_rate).first;
  const float *p_wave = wave.data_ptr<float>();
  int32_t num_samples = wave.numel();
  SHERPA_LOG(INFO) << "num_samples: " << num_samples;

  sherpa::RnntConvEmformerModel model(nn_model, device, false);
  TestStackUnstackStates(model, s);

  InitStream(model, s);

  int32_t chunk_length = model.ChunkLength();                 // 32
  int32_t right_context_length = model.RightContextLength();  // 8
  int32_t pad_length = model.PadLength();                     // 19
  int32_t chunk_length_pad = chunk_length + pad_length;
  int32_t context_size = model.ContextSize();

  SHERPA_LOG(INFO) << "chunk_length: " << chunk_length;
  SHERPA_LOG(INFO) << "right_context_length: " << right_context_length;
  SHERPA_LOG(INFO) << "pad_length: " << pad_length;

  int32_t k = 1600;  // feed this number of samples each time
  int32_t &num_processed_frames = s.GetNumProcessedFrames();

  for (int32_t c = 0; c < num_samples; c += k) {
    std::cout << "c: " << c << ", num_samples: " << num_samples
              << ", num_processed_frames: " << num_processed_frames << "\n";
    int32_t start = c;
    int32_t end = std::min(c + k, num_samples);
    s.AcceptWaveform(sampling_rate, wave.slice(/*dim*/ 0, start, end));
    int32_t num_frames_ready = s.NumFramesReady();
    while (num_frames_ready - num_processed_frames >= chunk_length_pad) {
      DecodeStream(model, s);
    }
  }
  s.InputFinished();
  // TODO(fangjun): Handle remaining frames
  auto &hyps = s.GetHyps();
  std::string text;
  for (int32_t i = 0; i != hyps.size(); ++i) {
    if (i < context_size) continue;
    text += sym[hyps[i]];
  }
  std::cout << "results:\n" << text << "\n";

  return 0;
}
