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

#ifndef SHERPA_CPP_API_WEBSOCKET_ONLINE_WEBSOCKET_SERVER_IMPL_H_
#define SHERPA_CPP_API_WEBSOCKET_ONLINE_WEBSOCKET_SERVER_IMPL_H_

#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>

#include "asio.hpp"
#include "sherpa/cpp_api/online_recognizer.h"
#include "sherpa/cpp_api/online_stream.h"
#include "sherpa/cpp_api/websocket/tee_stream.h"
#include "sherpa/csrc/parse_options.h"
#include "websocketpp/config/asio_no_tls.hpp"  // TODO(fangjun): support TLS
#include "websocketpp/server.hpp"

using server = websocketpp::server<websocketpp::config::asio>;
using connection_hdl = websocketpp::connection_hdl;

namespace sherpa {

struct OnlineWebsocketDecoderConfig {
  std::string nn_model;
  std::string tokens;

  /// Decoding method to use.
  /// Possible values are: greedy_search, modified_beam_search.
  std::string decoding_method = "greedy_search";

  /// Number of active paths in modified_beam_search.
  /// Used only when decoding_method is modified_beam_search.
  int32_t num_active_paths = 4;

  // All models from icefall are trained using audio data of
  // sample rate 16 kHz
  float sample_rate = 16000;

  bool use_gpu = false;
  int32_t max_batch_size = 5;

  float max_utterance_length = 100;  // seconds

  void Register(ParseOptions *po);
  void Validate() const;
};

struct OnlineWebsocketServerConfig {
  // assume you run it inside the ./build directory.
  std::string doc_root = "../sherpa/bin/web";  // root for the http server
  std::string log_file = "./log.txt";

  void Register(sherpa::ParseOptions *po);
  void Validate() const;
};

class OnlineWebsocketServer {
 public:
  explicit OnlineWebsocketServer(
      asio::io_context &io_conn,  // NOLINT
      asio::io_context &io_work,  // NOLINT
      const OnlineWebsocketServerConfig &config,
      const OnlineWebsocketDecoderConfig &decoder_config);

  void Run(uint16_t port);

 private:
  void SetupLog();

  // When a websocket client is connected, it will invoke this method
  // (Not for HTTP)
  void OnOpen(connection_hdl hdl);

  // Whena a websocket client is disconnected, it will invoke this method
  void OnClose(connection_hdl hdl);

  void OnMessage(connection_hdl hdl, server::message_ptr msg);

  // Close a websocket connection with given code and reason
  void Close(connection_hdl hdl, websocketpp::close::status::value code,
             const std::string &reason);

 private:
  asio::io_context &io_conn_;
  asio::io_context &io_work_;
  server server_;

  std::ofstream log_;
  sherpa::TeeStream tee_;

  OnlineWebsocketDecoderConfig decoder_config_;
  std::unique_ptr<OnlineRecognizer> recognizer_;
  std::map<connection_hdl, std::shared_ptr<OnlineStream>,
           std::owner_less<connection_hdl>>
      connections_;

  std::mutex mutex_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_WEBSOCKET_ONLINE_WEBSOCKET_SERVER_IMPL_H_
