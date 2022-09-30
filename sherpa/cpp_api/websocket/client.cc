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
    : hostname_(hostname), port_(port) {
      Connect();
      t_ = std::thread(&WebSocketClient::GetData, this);
    }

  void SendData(const void* data, size_t size) {
    ws_.binary(true);
    ws_.write(asio::buffer(data, size));
  }

  void GetData() {
    try {
      while (true) {
        beast::flat_buffer buffer;
        ws_.read(buffer);
        std::string message = beast::buffers_to_string(buffer.data());
        SHERPA_LOG(INFO) << message;
        json::object obj = json::parse(message).as_object();
        if (obj["status"] != "ok") {
          break;
        }
      }
    } catch (const beast::system_error & se) {
      // This indicates that the session was closed
      if (se.code() != websocket::error::closed) {
        SHERPA_LOG(ERROR) << se.code().message();
      }
    } catch (const std::exception & e) {
      SHERPA_LOG(ERROR) << e.what();
    }
  }

  ~WebSocketClient() {
    t_.join();
  }


 private:
  void Connect() {
    tcp::resolver resolver{ioc_};
    // Look up the domain name
    auto const results = resolver.resolve(hostname_, std::to_string(port_));
    // Make the connection on the IP address we get from a lookup
    auto ep = asio::connect(ws_.next_layer(), results);
    // Provide the value of the Host HTTP header during the WebSocket handshake.
    // See https://tools.ietf.org/html/rfc7230#section-5.4
    std::string host = hostname_ + ":" + std::to_string(ep.port());
    // Perform the websocket handshake
    ws_.handshake(host, "/");
  }

  std::string hostname_;
  int port_;
  asio::io_context ioc_;
  websocket::stream<tcp::socket> ws_{ioc_};
  std::thread t_;
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
  torch::Tensor wave_data = sherpa::ReadWave(wav_path, sample_rate).first;
  const int num_samples = wave_data.size(0);
  // send 0.32 second audio every time
  const float interval = 0.32;
  const int sample_interval = interval * sample_rate;
  for (int start = 0; start < num_samples; start += sample_interval) {
    int end = std::min(start + sample_interval, num_samples);
    std::vector<int16_t> data;
    data.reserve(end - start);
    for (int j = start; j < end; j++) {
      data.push_back(static_cast<int16_t>(wave_data[j].item<float>() * 32768));
    }
    // send PCM data with 16k1c16b format
    client.SendData(data.data(), data.size() * sizeof(int16_t));
    SHERPA_LOG(INFO) << "Send " << data.size() << " samples";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(interval * 1000 * 0.8)));
  }
  SHERPA_LOG(INFO) << "No more data by Client";
  return 0;
}
