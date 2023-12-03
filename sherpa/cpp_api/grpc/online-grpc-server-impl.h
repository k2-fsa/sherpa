// sherpa/cpp_api/websocket/online-grpc-server-impl.h
//
// Copyright (c)  2022  Xiaomi Corporation
//                2023  y00281951

#ifndef SHERPA_CPP_API_GRPC_ONLINE_GRPC_SERVER_IMPL_H_
#define SHERPA_CPP_API_GRPC_ONLINE_GRPC_SERVER_IMPL_H_

#include <deque>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <vector>
#include <string>

#include "asio.hpp"
#include "sherpa/cpp_api/online-recognizer.h"
#include "sherpa/cpp_api/online-stream.h"
#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/cpp_api/grpc/sherpa.grpc.pb.h"

namespace sherpa {
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;

struct Connection {
  // handle to the connection. We can use it to send messages to the client
  std::string reqid_ = "";
  std::shared_ptr<ServerReaderWriter<Response, Request>> stream_;
  std::shared_ptr<Request> request_;
  std::shared_ptr<Response> response_;
  std::shared_ptr<OnlineStream> s;

  int32_t nbest_ = 1;
  // The last time we received a message from the client
  // TODO(fangjun): Use it to disconnect from a client if it is inactive
  // for a specified time.
  std::chrono::steady_clock::time_point last_active;

  std::mutex mutex;  // protect sampels

  // Audio samples received from the client.
  //
  // The I/O threads receive audio samples into this queue
  // and invoke work threads to compute features
  std::deque<torch::Tensor> samples;

  bool start_flag_ = false;       // first time read request flag
  bool finish_flag_ = false;      // connection finish flag

  Connection() = default;
  Connection(std::shared_ptr<ServerReaderWriter<Response, Request>> stream,
             std::shared_ptr<Request> request,
             std::shared_ptr<Response> response,
             std::shared_ptr<OnlineStream> s)
             : stream_(stream),
               request_(request),
               response_(response),
               s(s),
               last_active(std::chrono::steady_clock::now()) {}
};

struct OnlineGrpcDecoderConfig {
  OnlineRecognizerConfig recognizer_config;

  // It determines how often the decoder loop runs.
  int32_t loop_interval_ms = 10;

  int32_t max_batch_size = 5;

  float padding_seconds = 0.8;

  void Register(ParseOptions *po);
  void Validate() const;
};

class OnlineGrpcServer;

class OnlineGrpcDecoder {
 public:
  /**
   * @param server  Not owned.
   */
  explicit OnlineGrpcDecoder(OnlineGrpcServer *server);

  // Compute features for a stream given audio samples
  void AcceptWaveform(std::shared_ptr<Connection> c);

  // signal that there will be no more audio samples for a stream
  void InputFinished(std::shared_ptr<Connection> c);

  void Run();

  OnlineGrpcDecoderConfig config_;
  std::map<std::string, std::shared_ptr<Connection>> connections_;
  std::unique_ptr<OnlineRecognizer> recognizer_;
  // It protects `connections_`, `ready_connections_`, and `active_`
  std::mutex mutex_;

 private:
  void ProcessConnections(const asio::error_code &ec);
  void SerializeResult(std::shared_ptr<Connection> c);
  void OnPartialResult(std::shared_ptr<Connection> c);
  void OnFinalResult(std::shared_ptr<Connection> c);
  void OnSpeechEnd(std::shared_ptr<Connection> c);
  /** It is called by one of the worker thread.
   */
  void Decode();

 private:
  OnlineGrpcServer *server_;  // not owned
  asio::steady_timer timer_;

  // Whenever a connection has enough feature frames for decoding, we put
  // it in this queue
  std::deque<std::shared_ptr<Connection>> ready_connections_;

  // If we are decoding a stream, we put it in the active_ set so that
  // only one thread can decode a stream at a time.
  std::set<std::string> active_;
};

struct OnlineGrpcServerConfig {
  OnlineGrpcDecoderConfig decoder_config;

  void Register(sherpa::ParseOptions *po);
  void Validate() const;
};

class OnlineGrpcServer final : public ASR::Service {
 public:
  OnlineGrpcServer(asio::io_context &io_work,  // NOLINT
                   const OnlineGrpcServerConfig &config);
  Status Recognize(ServerContext* context,
                   ServerReaderWriter<Response, Request>* reader) override;
  void Run();

  const OnlineGrpcServerConfig &GetConfig() const { return config_; }
  bool Contains(std::string reqid) const;
  asio::io_context &GetWorkContext() { return io_work_; }
  std::set<std::string> connections_;

 private:
  OnlineGrpcServerConfig config_;
  asio::io_context &io_work_;
  OnlineGrpcDecoder decoder_;

  mutable std::mutex mutex_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_GRPC_ONLINE_GRPC_SERVER_IMPL_H_
