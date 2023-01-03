// sherpa/cpp_api/websocket/offline-websocket-server-impl.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_WEBSOCKET_OFFLINE_WEBSOCKET_SERVER_IMPL_H_
#define SHERPA_CPP_API_WEBSOCKET_OFFLINE_WEBSOCKET_SERVER_IMPL_H_

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "asio.hpp"
#include "sherpa/cpp_api/offline-recognizer.h"
#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/cpp_api/websocket/http-server.h"
#include "sherpa/cpp_api/websocket/tee-stream.h"
#include "websocketpp/config/asio_no_tls.hpp"  // TODO(fangjun): support TLS
#include "websocketpp/server.hpp"

using server = websocketpp::server<websocketpp::config::asio>;
using connection_hdl = websocketpp::connection_hdl;

namespace sherpa {

struct ConnectionData {
  // Number of expected bytes sent from the client
  int32_t expected_byte_size = 0;

  // Number of bytes received so far
  int32_t cur = 0;

  // It saves the received contents from the client
  std::vector<int8_t> data;

  void Clear() {
    expected_byte_size = 0;
    cur = 0;
    data.clear();
  }
};
using ConnectionDataPtr = std::shared_ptr<ConnectionData>;

struct OfflineWebsocketDecoderConfig {
  OfflineRecognizerConfig recognizer_config;

  int32_t max_batch_size = 5;

  float max_utterance_length = 300;  // seconds

  void Register(ParseOptions *po);
  void Validate() const;
};

class OfflineWebsocketServer;

class OfflineWebsocketDecoder {
 public:
  /**
   * @param config Configuraion for the decoder.
   * @param server Borrowed from outside.
   */
  OfflineWebsocketDecoder(const OfflineWebsocketDecoderConfig &config,
                          OfflineWebsocketServer *server);

  /** Insert received data to the queue for decoding.
   *
   * @param hdl A handle to the connection. We can use it to send the result
   *            back to the client once it finishes decoding.
   * @param d  The received data
   */
  void Push(connection_hdl hdl, ConnectionDataPtr d);

  /** It is called by one of the work thread.
   */
  void Decode();

  const OfflineWebsocketDecoderConfig &GetConfig() const { return config_; }

 private:
  OfflineWebsocketDecoderConfig config_;

  /** When we have received all the data from the client, we put it into
   * this queue, the worker threads will get items from this queue for
   * decoding.
   *
   * Number of items to take from this queue is determined by
   * `--max-batch-size`. If there are not enough items in the queue, we won't
   * wait and take whatever we have for decoding.
   */
  std::mutex mutex_;
  std::deque<std::pair<connection_hdl, ConnectionDataPtr>> streams_;

  OfflineWebsocketServer *server_;  // Not owned
  OfflineRecognizer recognizer_;
};

struct OfflineWebsocketServerConfig {
  // assume you run it inside the ./build directory.
  std::string doc_root = "../sherpa/bin/web";  // root for the http server
  std::string log_file = "./log.txt";

  void Register(sherpa::ParseOptions *po);
  void Validate() const;
};

class OfflineWebsocketServer {
 public:
  OfflineWebsocketServer(asio::io_context &io_conn,  // NOLINT
                         asio::io_context &io_work,  // NOLINT
                         const OfflineWebsocketServerConfig &config,
                         const OfflineWebsocketDecoderConfig &decoder_config);

  asio::io_context &GetConnectionContext() { return io_conn_; }
  server &GetServer() { return server_; }

  void Run(uint16_t port);

 private:
  void SetupLog();

  // When a websocket client is connected, it will invoke this method
  // (Not for HTTP)
  void OnOpen(connection_hdl hdl);

  // Whena a websocket client is disconnected, it will invoke this method
  void OnClose(connection_hdl hdl);

  // When a HTTP client is connected, it will invoke this method
  void OnHttp(connection_hdl hdl);

  // When a message received from a websocket client, this method will
  // be invoked.
  //
  // The protocol between the client and the server is as follows:
  //
  // (1) The client connects to the server
  // (2) The client sends a binary message telling the server how many bytes
  //     it will send to the server. It contains 4-byte in little endian.
  // (3) The client sends a binary message containing the audio samples.
  //     If there are many audio samples, the client may split it into
  //     multiple binary messages.
  // (4) When the server receives all the samples from the client, it will
  //     start to decode them. Once decoded, the server sends a text message
  //     to the client containing the decoded results
  // (5) After receiving the decoded results from the server, if the client has
  //     another audio file to send, it repeats (2), (3), (4)
  // (6) If the client has no more audio files to decode, the client sends a
  //     text message containing "Done" to the server and closes the connection
  // (7) The server receives a text message "Done" and closes the connection
  //
  // Note:
  //  (a) All models in icefall are trained using audio samples at sampling
  //      rate 16 kHz. Please send audio samples with a sampling rate matching
  //      the one expected by the model.
  //  (b) All models in icefall use features extracted from audio samples
  //      normalized to the range [-1, 1]. Please send normalized audio samples
  //      if you use models from icefall.
  //  (c) Only sound files with a single channel is supported
  //  (d) Step (2) and step (3) can be merged into one step to send bandwidth.
  //  (e) Only audio samples are sent. For instance, if we want to decode
  //      a WAVE file, the header of the WAVE is not sent.
  void OnMessage(connection_hdl hdl, server::message_ptr msg);

  // Close a websocket connection with given code and reason
  void Close(connection_hdl hdl, websocketpp::close::status::value code,
             const std::string &reason);

 private:
  asio::io_context &io_conn_;
  asio::io_context &io_work_;
  HttpServer http_server_;
  server server_;

  std::map<connection_hdl, ConnectionDataPtr, std::owner_less<connection_hdl>>
      connections_;
  std::mutex mutex_;

  OfflineWebsocketServerConfig config_;

  std::ofstream log_;
  sherpa::TeeStream tee_;

  OfflineWebsocketDecoder decoder_;
  int32_t max_byte_size_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_WEBSOCKET_OFFLINE_WEBSOCKET_SERVER_IMPL_H_
