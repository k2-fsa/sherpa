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

#include "sherpa/cpp_api/websocket/microphone.h"

#include <stdio.h>

#include <utility>

#include "portaudio.h"  // NOLINT
#include "torch/script.h"

namespace sherpa {

static int RecordCallback(const void *input_buffer, void * /*output_buffer*/,
                          unsigned long frames_per_buffer,  // NOLINT
                          const PaStreamCallbackTimeInfo * /*time_info*/,
                          PaStreamCallbackFlags /*status_flags*/,
                          void *user_data) {
  Microphone *mic = reinterpret_cast<Microphone *>(user_data);

  auto samples =
      torch::from_blob(static_cast<float *>(const_cast<void *>(input_buffer)),
                       {static_cast<int>(frames_per_buffer)}, torch::kFloat);

  mic->InvokeCallback(samples);

  // return stop ? paComplete : paContinue;
  return paContinue;
}

Microphone::Microphone() {
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }
}

Microphone::~Microphone() {
  PaError err = paNoError;

  if (stream_) {
    err = Pa_CloseStream(stream_);
    if (err != paNoError) {
      fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
      exit(EXIT_FAILURE);
    }
  }

  err = Pa_Terminate();
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }
}

void Microphone::StartMicrophone(std::function<void(torch::Tensor)> callback) {
  callback_ = std::move(callback);

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

  PaError err =
      Pa_OpenStream(&stream_, &param, nullptr, /* &outputParameters, */
                    sample_rate_,
                    0,          // frames per buffer
                    paClipOff,  // we won't output out of range samples
                                // so don't bother clipping them
                    RecordCallback, this);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  err = Pa_StartStream(stream_);
  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }
}

}  // namespace sherpa
