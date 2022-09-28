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

#include <iostream>
#include <memory>
#include <string>
#include <thread>
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

  namespace beast = boost::beast;
  namespace http = beast::http;
  namespace websocket = beast::websocket;
  namespace asio = boost::asio;
  namespace json = boost::json;
  using tcp = boost::asio::ip::tcp;

  class ConnectionHandler {
    public:
      ConnectionHandler(tcp::socket&& socket,
          std::shared_ptr<sherpa::OnlineAsr> online_asr) : 
        ws_(std::move(socket)), online_asr_(std::move(online_asr)) {}

      void operator()() {
        try {
          ws_.accept();
          auto recog_stream = online_asr_->CreateStream();

          while (recog_stream && continuous_decoding_) {
            // get stream data
            // audio input for model should be 16k1c16b
            beast::flat_buffer buffer;
            ws_.read(buffer);
            if (buffer.size() == 0) {
              // should close socket when no more data
              continuous_decoding_ = false;
            }
            int num_samples = buffer.size() / sizeof(int16_t);
            const int16_t* pcm_data = static_cast<const int16_t*>(buffer.data().data());
            auto wav_stream_tensor = torch::from_blob(const_cast<int16_t *>(pcm_data),
                {num_samples}, torch::kInt16).to(torch::kFloat) / 32768;
            recog_stream->AcceptWaveform(16000, wav_stream_tensor);

            // decoding stream
            if (online_asr_->IsReady(recog_stream.get())) {
              online_asr_->DecodeStream(recog_stream.get());
              std::string transcript = online_asr_->GetResult(recog_stream.get());
              std::string endpoint_type = "endpoint_inactive";

              // reset stream when Endpoint active
              if (recog_stream->IsEndpoint()) {
                transcript += " ";
                endpoint_type = "endpoint_active";
                recog_stream = online_asr_->CreateStream();
              }
              // update result
              json::value rv = {{"status", "ok"},
                {"type", endpoint_type},
                {"nbest", transcript}};
              ws_.text(true);
              ws_.write(asio::buffer(json::serialize(rv)));
            }
          }
        } catch (beast::system_error const& se) {
          SHERPA_LOG(INFO) << se.code().message();
          // This indicates that the session was closed
          if (se.code() == websocket::error::closed) {
          }
        } catch (std::exception const& e) {
          SHERPA_LOG(WARNING) << e.what();
          ws_.close(websocket::close_code::normal);
        }
      }

    private:
      websocket::stream<tcp::socket> ws_;
      std::shared_ptr<sherpa::OnlineAsr> online_asr_ = nullptr;
      bool continuous_decoding_ = true;
  };

  class WebSocketServer {
    public:
      WebSocketServer(int port, const sherpa::OnlineAsrOptions opts) 
        : port_(port), 
        online_asr_(std::make_shared<sherpa::OnlineAsr>(opts)) {}

      void Start() {
        try {
          auto const address = asio::ip::make_address("0.0.0.0");
          tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port_)}};
          while (true) {
            // This will receive the new connection
            tcp::socket socket{ioc_};
            // Block until we get a connection
            acceptor.accept(socket);
            // Launch the session, transferring ownership of the socket
            ConnectionHandler handler(std::move(socket), online_asr_);
            std::thread t(std::move(handler));
            t.detach();
          }
        } catch (const std::exception& e) {
          SHERPA_LOG(FATAL) << e.what();
        }
      }

    private:
      int port_;
      // The io_context is required for all I/O
      asio::io_context ioc_{1};
      std::shared_ptr<sherpa::OnlineAsr> online_asr_ = nullptr;
  };

}  // namespace sherpa

#endif  //SHERPA_CPP_API_WEBSOCKET_SERVER_H_
