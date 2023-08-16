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
#include "sherpa/cpp_api/online-recognizer.h"
#include "sherpa/cpp_api/online-stream.h"
#include "sherpa/csrc/fbank-features.h"

static constexpr const char *kUsageMessage = R"(
Online (streaming) automatic speech recognition with sherpa.

Usage:
(1) View help information.

  sherpa-online-microphone --help

(2) Use a pretrained model for recognition

  sherpa-online-microphone \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --decoding-method=greedy_search

To use fast_beam_search with an LG, use

  sherpa-online-microphone \
    --decoding-method=fast_beam_search \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --lg=/path/to/LG.pt \
    --use-gpu=false

(3) To use an LSTM model for recognition

  sherpa-online-microphone \
    --encoder-model=/path/to/encoder_jit_trace.pt \
    --decoder-model=/path/to/decoder_jit_trace.pt \
    --joiner-model=/path/to/joiner_jit_trace.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false

(4) To use a streaming Zipformer model for recognition

  sherpa-online-microphone
    --encoder-model=/path/to/encoder_jit_trace.pt \
    --decoder-model=/path/to/decoder_jit_trace.pt \
    --joiner-model=/path/to/joiner_jit_trace.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false

See
https://k2-fsa.github.io/sherpa/sherpa/pretrained_models/online_transducer.html
for more details.
)";

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
  signal(SIGINT, Handler);

  if (argc == 1) {
    fprintf(stderr, "%s\n", kUsageMessage);
    exit(0);
  }

  Microphone mic;

  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  sherpa::ParseOptions po(kUsageMessage);
  sherpa::OnlineRecognizerConfig config;
  config.Register(&po);

  po.Read(argc, argv);
  if (argc == 0 || po.NumArgs() != 0) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  config.Validate();

  float sample_rate = 16000;
  if (config.feat_config.fbank_opts.frame_opts.samp_freq != sample_rate) {
    std::cerr
        << "The model was trained using training data with sample rate 16000. "
        << "We don't support resample yet\n";
    exit(EXIT_FAILURE);
  }

  sherpa::OnlineRecognizer recognizer(config);

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
      if (static_cast<int32_t>(result.size()) != result_len) {
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
