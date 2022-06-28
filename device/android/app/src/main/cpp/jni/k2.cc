/**
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
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

#include <android/log.h>
#include <jni.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "sherpa/csrc/rnnt_beam_search.h"
#include "sherpa/decode_stream.h"
#include "torch/all.h"
#include "torch/script.h"
#include "torch/utils.h"

#define TAG "k2"

// Use android logging tool, so we can see the log in logcat.
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

namespace sherpa {
namespace jni {

// TODO: Move these global variables to java size.
std::shared_ptr<RnntConformerModel> g_model;
std::shared_ptr<DecodeStream> g_decode_stream;
std::unordered_map<int, std::string> g_token_dict;

// TODO: Make them configurable
static int g_left_context = 32;
static int g_chunk_size = 8;
static int g_right_context = 0;

void Init(JNIEnv *env, jobject, jstring jModelPath, jstring jTokenPath) {
  torch::Device device(torch::kCPU);
  const char *pModelPath = (env)->GetStringUTFChars(jModelPath, nullptr);
  std::string modelPath = std::string(pModelPath);
  LOGI("model path: %s\n", modelPath.c_str());

  RnntConformerModel model(modelPath);
  g_model = std::make_shared<RnntConformerModel>(model);

  const char *pTokenPath = (env)->GetStringUTFChars(jTokenPath, nullptr);
  std::string tokenPath = std::string(pTokenPath);
  LOGI("token path : %s\n", tokenPath.c_str());

  g_token_dict.clear();
  std::ifstream ifile(tokenPath.c_str());
  std::string line;
  while (std::getline(ifile, line)) {
    std::istringstream iss(line);
    int32_t id;
    std::string token;
    if (!(iss >> token >> id)) {
      break;
    }
    g_token_dict.insert(std::pair<int32_t, std::string>(id, token));
  }
  LOGI("token dict size : %lu\n", g_token_dict.size());
}

void InitDecodeStream() {
  int32_t context_size = g_model->ContextSize();
  int32_t blank_id = g_model->BlankId();

  auto initial_states = g_model->GetEncoderInitStates(g_left_context);

  // we can do like this because the batch size is 1.
  auto tokens = std::vector<int64_t>(context_size, blank_id);
  auto decoder_input = torch::from_blob(
      tokens.data(), {1, static_cast<int64_t>(tokens.size())}, torch::kInt64);

  auto initial_decoder_out = g_model->ForwardDecoder(decoder_input);

  initial_decoder_out =
      g_model->ForwardDecoderProj(initial_decoder_out.squeeze(1));

  auto decode_stream =
      DecodeStream(initial_states, initial_decoder_out, context_size, blank_id);
  g_decode_stream = std::make_shared<DecodeStream>(decode_stream);
}

void AcceptWaveform(JNIEnv *env, jobject, jfloatArray jWaveform) {
  auto device = torch::Device(torch::kCPU);
  jsize size = env->GetArrayLength(jWaveform);
  std::vector<float> waveform(size);
  env->GetFloatArrayRegion(jWaveform, 0, size, &waveform[0]);
  auto wave_data =
      torch::from_blob(waveform.data(), {size}, torch::kFloat).to(device);
  g_decode_stream->AcceptWaveform(wave_data);
}

bool IsFinished(JNIEnv *env, jobject) { return g_decode_stream->IsFinished(); }

void InputFinished() {
  g_decode_stream->InputFinished();
  g_decode_stream->AddTailPaddings(
      (2 + g_right_context) * g_model->SubSamplingFactor() + 7);
}

jstring Decode(JNIEnv *env, jobject) {
  int subsampling_factor = g_model->SubSamplingFactor();
  int chunk_size =
      (2 + g_chunk_size + g_right_context) * subsampling_factor + 3;
  int chunk_shift = g_chunk_size * subsampling_factor;

  torch::Tensor features = g_decode_stream->GetFeature(chunk_size, chunk_shift);
  features = features.unsqueeze(0);

  torch::Tensor features_length =
      torch::tensor(std::vector<int32_t>(1, features.size(1)),
                    torch::device(features.device()));

  auto states = g_decode_stream->GetState();
  states = {states[0].unsqueeze(2), states[1].unsqueeze(2)};

  torch::Tensor processed_frames = torch::tensor(
      std::vector<int64_t>(1, g_decode_stream->GetNumProcessedFrames()),
      torch::device(features.device()));

  auto encoder_out_tuple = g_model->StreamingForwardEncoder(
      features, features_length, states, processed_frames, g_left_context,
      g_right_context);
  torch::Tensor encoder_out = std::get<0>(encoder_out_tuple);
  torch::Tensor encoder_out_length = std::get<1>(encoder_out_tuple);
  auto next_states = std::get<2>(encoder_out_tuple);
  next_states = {next_states[0].squeeze(2), next_states[1].squeeze(2)};

  torch::Tensor decoder_out = g_decode_stream->GetDecoderOut();
  auto hyp = g_decode_stream->GetHyp();
  std::vector<std::vector<int32_t>> hyps = {hyp};

  auto next_decoder_out =
      StreamingGreedySearch(*g_model, encoder_out, decoder_out, &hyps);

  g_decode_stream->SetDecoderOut(next_decoder_out);
  g_decode_stream->SetHyp(hyps[0]);
  g_decode_stream->SetState(next_states);
  g_decode_stream->UpdateNumProcessedFrames(
      encoder_out_length[0].item<int32_t>());

  int32_t context_size = g_model->ContextSize();
  hyp = g_decode_stream->GetHyp();
  std::ostringstream oss;
  for (int32_t i = context_size; i < hyp.size(); ++i) {
    oss << g_token_dict[hyp[i]];
  }
  std::string text = oss.str();
  text = std::regex_replace(text, std::regex("▁"), " ");
  return env->NewStringUTF(text.c_str());
}
}  // namespace jni
}  // namespace sherpa

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  jclass c = env->FindClass("com/xiaomi/k2/Recognizer");
  if (c == nullptr) {
    return JNI_ERR;
  }

  static const JNINativeMethod methods[] = {
      {"init", "(Ljava/lang/String;Ljava/lang/String;)V",
       reinterpret_cast<void *>(sherpa::jni::Init)},
      {"initDecodeStream", "()V",
       reinterpret_cast<void *>(sherpa::jni::InitDecodeStream)},
      {"acceptWaveform", "([F)V",
       reinterpret_cast<void *>(sherpa::jni::AcceptWaveform)},
      {"inputFinished", "()V",
       reinterpret_cast<void *>(sherpa::jni::InputFinished)},
      {"decode", "()Ljava/lang/String;",
       reinterpret_cast<void *>(sherpa::jni::Decode)},
      {"isFinished", "()Z", reinterpret_cast<void *>(sherpa::jni::IsFinished)},
  };
  int rc = env->RegisterNatives(c, methods,
                                sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
