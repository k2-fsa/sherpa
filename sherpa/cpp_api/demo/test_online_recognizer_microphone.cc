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
#include <signal.h>

#include "portaudio.h"  // NOLINT
#include "sherpa/cpp_api/online_recognizer.h"
#include "sherpa/cpp_api/online_stream.h"
#include "sherpa/csrc/fbank_features.h"

class Microphone {
 public:
  Microphone() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
      fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
      exit(-1);
    }
  }
  ~Microphone() {
    PaError err = Pa_Terminate();
    if (err != paNoError) {
      fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
      exit(-1);
    }
  }
};

bool stop = false;

static int RecordCallback(const void *input_buffer, void * /*output_buffer*/,
                          unsigned long frames_per_buffer,  // NOLINT
                          const PaStreamCallbackTimeInfo * /*time_info*/,
                          PaStreamCallbackFlags /*status_flags*/,
                          void *user_data) {
  auto s = reinterpret_cast<sherpa::OnlineStream *>(user_data);
  auto samples =
      torch::from_blob(static_cast<float *>(const_cast<void *>(input_buffer)),
                       {static_cast<int>(frames_per_buffer)}, torch::kFloat);

  s->AcceptWaveform(16000, samples);

  return stop ? paComplete : paContinue;
}
static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nexiting...\n");
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  if (argc != 3) {
    const char *msg =
        "Usage: ./bin/test_online_recognizer_microphone /path/to/nn_model "
        "/path/to/tokens.txt \n";
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
  }

  signal(SIGINT, Handler);
  Microphone mic;

  std::string nn_model = argv[1];
  std::string tokens = argv[2];
  float sample_rate = 16000;
  bool use_gpu = false;

  sherpa::DecodingOptions opts;
  opts.method = sherpa::kGreedySearch;
  sherpa::OnlineRecognizer recognizer(nn_model, tokens, opts, use_gpu,
                                      sample_rate);

  auto s = recognizer.CreateStream();

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "num devices: %d\n", num_devices);

  PaStreamParameters param;

  param.device = Pa_GetDefaultInputDevice();
  if (param.device == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Use default device: %d\n", param.device);

  const PaDeviceInfo *info = Pa_GetDeviceInfo(param.device);
  fprintf(stderr, "  Name: %s\n", info->name);
  fprintf(stderr, "  Max input channels: %d\n", info->maxInputChannels);

  param.channelCount = 1;
  param.sampleFormat = paFloat32;

  param.suggestedLatency = info->defaultLowInputLatency;
  param.hostApiSpecificStreamInfo = nullptr;

  PaStream *stream;
  PaError err =
      Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                    sample_rate,
                    0,          // frames per buffer
                    paClipOff,  // we won't output out of range samples
                                // so don't bother clipping them
                    RecordCallback, s.get());
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  err = Pa_StartStream(stream);
  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  int32_t result_len = 0;
  while (!stop) {
    if (recognizer.IsReady(s.get())) {
      recognizer.DecodeStream(s.get());
      auto result = recognizer.GetResult(s.get()).text;
      if (result.size() != result_len) {
        result_len = result.size();
        fprintf(stderr, "%s\n", result.c_str());
      }
    }

    Pa_Sleep(20);  // sleep for 20ms
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
