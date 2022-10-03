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
#include <string>
#include <thread> // NOLINT

#include "boost/asio/connect.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/websocket.hpp"
#include "boost/json/src.hpp"

#include "sherpa/csrc/fbank_features.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/parse_options.h"


namespace asio = boost::asio;
using tcp = boost::asio::ip::tcp;
namespace beast = boost::beast;
namespace json = boost::json;
namespace http = beast::http;
namespace websocket = beast::websocket;

class WebSocketClient {
 public:
  WebSocketClient(const std::string& hostname, int port)
    : hostname_(hostname), port_(port), alive_(true) {
      Open();
      get_thread_ = std::thread(&WebSocketClient::Get, this);
    }

  void Put(const void* data, size_t size) {
    ws_.binary(true);
    ws_.write(asio::buffer(data, size));
  }

  void Get() {
    try {
      while (alive_) {
        beast::flat_buffer buffer;
        ws_.read(buffer);
        std::string message = beast::buffers_to_string(buffer.data());
        SHERPA_LOG(INFO) << message;
        json::object obj = json::parse(message).as_object();
        if (obj["status"] != "ok") {
          alive_ = false;
        }
      }
      Close();
    } catch (const beast::system_error & se) {
      if (se.code() != websocket::error::closed) {
        SHERPA_LOG(WARNING) << se.code().message();
      }
    } catch (const std::exception & e) {
      SHERPA_LOG(WARNING) << e.what();
    }
  }

  ~WebSocketClient() {
    get_thread_.join();
  }

 private:
  void Open() {
    tcp::resolver resolver{ioc_};
    // Make IP address get from a domain name lookup
    auto const results = resolver.resolve(hostname_, std::to_string(port_));
    auto ep = asio::connect(ws_.next_layer(), results);
    // Provide the value of the Host HTTP header during the WebSocket handshake.
    // See https://tools.ietf.org/html/rfc7230#section-5.4
    std::string host = hostname_ + ":" + std::to_string(ep.port());
    // Make WebSocket handshake
    ws_.handshake(host, "/");
  }

  void Close() { ws_.close(websocket::close_code::normal); }

  std::string hostname_;
  int port_;
  asio::io_context ioc_;
  websocket::stream<tcp::socket> ws_{ioc_};
  std::thread get_thread_;
  bool alive_;
};

static constexpr const char *kUsageMessage = R"(./bin/websocket-client --server-ip=127.0.0.1 --server-port=6006 --wav-path=test.wav)";

int main(int argc, char* argv[]) {
  sherpa::ParseOptions po(kUsageMessage);
  std::string ip;
  int port;
  std::string wav_path;
  po.Register("server-ip", &ip, "Server ip to connect");
  po.Register("server-port", &port, "Server port to connect");
  po.Register("wav-path", &wav_path, "path of test wav");
  po.Read(argc, argv);
  if (argc < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  WebSocketClient client(ip, port);

  const int sample_rate = 16000;
  torch::Tensor tail_padding =
    torch::zeros({static_cast<int32_t>(0.32 * sample_rate)}, torch::kFloat);
  torch::Tensor wave_data = sherpa::ReadWave(wav_path, sample_rate).first;
  const int num_samples = wave_data.size(0);
  // send 0.32 second audio every time
  const float interval = 0.32;
  const int sample_interval = interval * sample_rate;
  size_t tot_send_samples = 0;
  for (int start = 0; start < num_samples; start += sample_interval) {
    int end = std::min(start + sample_interval, num_samples);
    std::vector<int16_t> data;
    data.reserve(end - start);
    for (int j = start; j < end; j++) {
      data.push_back(static_cast<int16_t>(wave_data[j].item<float>() * 32768));
    }
    if (end == num_samples) {
      for (int j = 0; j < tail_padding.size(0); j++) {
        data.push_back(static_cast<int16_t>(tail_padding[j].item<float>()));
      }
    }
    // send PCM data with 16k1c16b format
    client.Put(data.data(), data.size() * sizeof(int16_t));
    // send a empty package to tell Server end when no more input
    if (end == num_samples) {
      std::vector<int16_t> fake_data;
      client.Put(fake_data.data(), fake_data.size() * sizeof(int16_t));
    }
    tot_send_samples += data.size();
    SHERPA_LOG(INFO) << "Cur Send " << data.size() << " samples"
      << ", tot already Send " << tot_send_samples << " samples";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(interval * 1000 * 0.8)));
  }

  SHERPA_LOG(INFO) << "Client has no more data, should exit";
  return 0;
}
