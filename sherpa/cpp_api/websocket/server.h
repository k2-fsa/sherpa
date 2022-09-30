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
#ifndef SHERPA_CPP_API_WEBSOCKET_SERVER_H_
#define SHERPA_CPP_API_WEBSOCKET_SERVER_H_
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
    online_asr_(std::move(online_asr)),
    alive_(true) {
      last_active_time_ = std::chrono::system_clock::now();
      detect_alive_ = std::thread(
            &ConnectionHandler::DetectAlive, this);
      detect_alive_.detach();
    }

  void DetectAlive() {
    while (alive_) {
      std::chrono::milliseconds timespan(10000);
      std::this_thread::sleep_for(timespan);
      std::chrono::duration<double> elapsed_seconds
        = std::chrono::system_clock::now() - last_active_time_;
      if (elapsed_seconds.count() > idle_timeout_) {
        alive_ = false;
        SHERPA_LOG(INFO) << "idle_timeout=" << idle_timeout_
          << " active, will close the socket.";
      }
    }
  }

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
          std::string transcript = online_asr_->GetResult(
              decode_stream.get());
          std::string endpoint_type = "endpoint_inactive";

          online_asr_->DecodeStream(decode_stream.get());
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
        last_active_time_ = std::chrono::system_clock::now();
      }
    } catch (const beast::system_error & se) {
      SHERPA_LOG(INFO) << se.code().message();
    } catch (const std::exception & e) {
      SHERPA_LOG(WARNING) << e.what();
    }
    alive_ = false;
    ws_.close(websocket::close_code::normal);
  }

 private:
  websocket::stream<tcp::socket> ws_;
  std::shared_ptr<sherpa::OnlineAsr> online_asr_ = nullptr;
  std::thread detect_alive_;
  std::chrono::system_clock::time_point last_active_time_;
  // how long to keep socket from last active
  const uint64_t idle_timeout_ = 600;
  bool alive_ = true;
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

#endif  // SHERPA_CPP_API_WEBSOCKET_SERVER_H_
