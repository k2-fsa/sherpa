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
#ifndef SHERPA_CPP_API_WEBSOCKET_MICROPHONE_H_
#define SHERPA_CPP_API_WEBSOCKET_MICROPHONE_H_

#include <stdio.h>

#include "portaudio.h"  // NOLINT
#include "torch/script.h"
#include "websocketpp/client.hpp"
#include "websocketpp/config/asio_no_tls_client.hpp"

using client = websocketpp::client<websocketpp::config::asio_client>;

namespace sherpa {

class Microphone {
 public:
  Microphone();
  ~Microphone();
  Microphone(const Microphone &) = delete;
  Microphone &operator=(const Microphone &) = delete;

  /**
   * @param callback  When there is data available, the passed callback is
   *                  invoked.
   */
  // void StartMicrophone(std::function<void(torch::Tensor)> callback);
  void _StartMicrophone();
  void StartMicrophone(client *c, websocketpp::connection_hdl hdl) {
    c_ = c;
    hdl_ = hdl;

    t_ = std::thread([&]() { _StartMicrophone(); });
  }

  void Push(torch::Tensor samples);

  client *c_;
  websocketpp::connection_hdl hdl_;
  std::thread t_;

 private:
  torch::Tensor samples_;
  std::function<void(torch::Tensor)> callback_;
  PaStream *stream_ = nullptr;

  float sample_rate_ = 16000;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_WEBSOCKET_MICROPHONE_H_
