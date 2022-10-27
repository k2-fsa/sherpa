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

#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <utility>

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

  void Register(ParseOptions *po);
  void Validate() const;
};

class OnlineWebsocketServer;

class OnlineWebsocketDecoder {
 public:
  /**
   * @param config  Configuration for the decoder.
   * @param server  Not owned.
   */
  OnlineWebsocketDecoder(const OnlineWebsocketDecoderConfig &config,
                         OnlineWebsocketServer *server);

  OnlineRecognizer *GetRecognizer() { return recognizer_.get(); }
  const OnlineWebsocketDecoderConfig &GetConfig() const { return config_; }

  void Push(connection_hdl hdl, std::shared_ptr<OnlineStream> s);

  /** It is called by one of the worker thread.
   */
  void Decode();

 private:
  std::unique_ptr<OnlineRecognizer> recognizer_;
  OnlineWebsocketDecoderConfig config_;
  OnlineWebsocketServer *server_;  // not owned

  std::mutex mutex_;
  std::deque<std::pair<connection_hdl, std::shared_ptr<OnlineStream>>> streams_;
  std::set<OnlineStream *> active_;
};

struct OnlineWebsocketServerConfig {
  std::string log_file = "./log.txt";

  void Register(sherpa::ParseOptions *po);
};

class OnlineWebsocketServer {
 public:
  explicit OnlineWebsocketServer(
      asio::io_context &io_conn,  // NOLINT
      asio::io_context &io_work,  // NOLINT
      const OnlineWebsocketServerConfig &config,
      const OnlineWebsocketDecoderConfig &decoder_config);

  void Run(uint16_t port);

  asio::io_context &GetConnectionContext() { return io_conn_; }
  asio::io_context &GetWorkContext() { return io_work_; }
  server &GetServer() { return server_; }

  void Send(connection_hdl hdl, const std::string &text);

 private:
  void SetupLog();

  // When a websocket client is connected, it will invoke this method
  // (Not for HTTP)
  void OnOpen(connection_hdl hdl);

  // When a websocket client is disconnected, it will invoke this method
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

  OnlineWebsocketDecoder decoder_;
  std::map<connection_hdl, std::shared_ptr<OnlineStream>,
           std::owner_less<connection_hdl>>
      connections_;

  std::mutex mutex_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_WEBSOCKET_ONLINE_WEBSOCKET_SERVER_IMPL_H_
