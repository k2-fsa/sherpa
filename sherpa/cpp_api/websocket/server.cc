/**
 * Copyright      2022  (authors: Pingfeng Luo)
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
#include <chrono> // NOLINT
#include <iostream>
#include <memory>
#include <ratio> // NOLINT
#include <string>
#include <thread> // NOLINT
#include <utility>

#include "boost/asio/connect.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/websocket.hpp"
#include "boost/json/src.hpp"

#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_asr.h"
#include "sherpa/csrc/parse_options.h"

namespace sherpa {
namespace asio = boost::asio;
using tcp = boost::asio::ip::tcp;
namespace beast = boost::beast;
namespace json = boost::json;
namespace http = beast::http;
namespace websocket = beast::websocket;

class ConnectionHandler {
 public:
  ConnectionHandler(tcp::socket&& socket,
      std::shared_ptr<sherpa::OnlineAsr> online_asr) :
    ws_(std::move(socket)),
    alive_(true),
    online_asr_(std::move(online_asr)) {}

  void operator()() {
    try {
      ws_.accept();
      auto decode_stream = online_asr_->CreateStream();
      while (decode_stream && alive_) {
        // get PCM data with 16k1c16b format
        beast::flat_buffer buffer;
        ws_.read(buffer);
        int num_samples = buffer.size() / sizeof(int16_t);
        const int16_t* pcm_data
          = static_cast<const int16_t*>(buffer.data().data());
        auto wav_stream_tensor = torch::from_blob(
            const_cast<int16_t *>(pcm_data),
            {num_samples},
            torch::kInt16).to(torch::kFloat) / 32768;

        // decode stream
        decode_stream->AcceptWaveform(16000, wav_stream_tensor);
        if (online_asr_->IsReady(decode_stream.get())) {
          online_asr_->DecodeStream(decode_stream.get());

          std::string transcript = online_asr_->GetResult(
              decode_stream.get());
          std::string endpoint_type = "endpoint_inactive";
          if (decode_stream->IsEndpoint()) {
            endpoint_type = "endpoint_active";
            // reset stream when Endpoint active
            decode_stream = online_asr_->CreateStream();
          }
          // update result
          json::value rv = {{"status", "ok"},
            {"type", endpoint_type},
            {"nbest", transcript}};
          ws_.text(true);
          ws_.write(asio::buffer(json::serialize(rv)));
        }

        // end section when no more input
        if (num_samples == 0) {
          json::value rv = {{"status", "end"}};
          ws_.text(true);
          ws_.write(asio::buffer(json::serialize(rv)));
          alive_ = false;
        }
      }
    } catch (const beast::system_error & se) {
      SHERPA_LOG(INFO) << se.code().message();
    } catch (const std::exception & e) {
      SHERPA_LOG(WARNING) << e.what();
      ws_.close(websocket::close_code::normal);
    }
  }

 private:
  websocket::stream<tcp::socket> ws_;
  bool alive_ = true;
  std::shared_ptr<sherpa::OnlineAsr> online_asr_ = nullptr;
};

class WebSocketServer {
 public:
  WebSocketServer(int port, const sherpa::OnlineAsrOptions opts) :
    online_asr_(std::make_shared<sherpa::OnlineAsr>(opts)) {
      StartServer(port);
    }

 private:
  void StartServer(int port) {
    try {
      auto const address = asio::ip::make_address("0.0.0.0");
      tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port)}};
      while (true) {
        tcp::socket socket{ioc_};
        // Block until a new connection
        acceptor.accept(socket);
        // Start a new thread to handle the new connection
        ConnectionHandler handler(std::move(socket), online_asr_);
        std::thread t(std::move(handler));
        t.detach();
      }
    } catch (const std::exception& e) {
      SHERPA_LOG(FATAL) << e.what();
    }
  }

  // The io_context for all I/O
  asio::io_context ioc_{1};
  std::shared_ptr<sherpa::OnlineAsr> online_asr_ = nullptr;
};

}  // namespace sherpa


static constexpr const char *kUsageMessage = R"(
Online (streaming) automatic speech recognition RPC server with sherpa.

Usage:
(1) View help information.

  ./bin/websocket-server --help

(2) Run server

  ./bin/websocket-server \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --server-port=6006
)";

int main(int argc, char *argv[]) {
  // set torch
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;
  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  // set OnlineAsr option
  sherpa::ParseOptions po(kUsageMessage);
  sherpa::OnlineAsrOptions opts;
  opts.Register(&po);
  int port;
  po.Register("server-port", &port, "Server port to listen on");
  po.Read(argc, argv);
  if (argc < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }
  SHERPA_LOG(INFO) << "decoding method: " << opts.decoding_method;
  opts.Validate();
  // tips : trailing_silence for EndpointConfig is after sampling
  opts.endpoint_config.rule1 = sherpa::EndpointRule(false, 0.8, 0.0);
  opts.endpoint_config.rule3 = sherpa::EndpointRule(true, 0.4, 2.0);
  opts.endpoint_config.rule3 = sherpa::EndpointRule(false, 0.0, 20);

  SHERPA_LOG(INFO) << "ASR Server Listening at port " << port;
  sherpa::WebSocketServer server(port, opts);
  return 0;
}
