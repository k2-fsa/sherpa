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
#include <string>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/cpp_api/websocket/microphone.h"
#include "sherpa/csrc/log.h"
#include "torch/script.h"
#include "websocketpp/client.hpp"
#include "websocketpp/config/asio_no_tls_client.hpp"
#include "websocketpp/uri.hpp"

using client = websocketpp::client<websocketpp::config::asio_client>;

using message_ptr = client::message_ptr;
using websocketpp::connection_hdl;

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa using websocket.

Usage:

./bin/sherpa-online-websocket-client-microphone --help

./bin/sherpa-online-websocket-client-microphone \
  --server-ip=127.0.0.1 \
  --server-port=6006
)";

static void OnMessage(client *c, connection_hdl hdl, message_ptr msg) {
  static std::string last;
  const std::string &payload = msg->get_payload();
  if (payload == "Done") {
    websocketpp::lib::error_code ec;
    c->close(hdl, websocketpp::close::status::normal, "I'm exiting now", ec);
    if (ec) {
      std::cerr << "Failed to close\n";
      exit(EXIT_FAILURE);
    }
  } else if (payload.size() != last.size() || payload != last) {
    SHERPA_LOG(INFO) << payload;
    last = payload;
  }
}

int32_t main(int32_t argc, char *argv[]) {
  std::string server_ip = "127.0.0.1";
  int32_t server_port = 6006;

  sherpa::ParseOptions po(kUsageMessage);

  po.Register("server-ip", &server_ip, "IP address of the websocket server");
  po.Register("server-port", &server_port, "Port of the websocket server");

  if (argc == 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  po.Read(argc, argv);

  if (!websocketpp::uri_helper::ipv4_literal(server_ip.begin(),
                                             server_ip.end())) {
    SHERPA_LOG(FATAL) << "Invalid server IP: " << server_ip;
  }

  if (server_port <= 0 || server_port > 65535) {
    SHERPA_LOG(FATAL) << "Invalid server port: " << server_port;
  }

  if (po.NumArgs() != 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  bool secure = false;
  std::string resource = "/";
  websocketpp::uri uri(secure, server_ip, server_port, resource);

  client c;

  c.clear_access_channels(websocketpp::log::alevel::all);
  c.set_access_channels(websocketpp::log::alevel::connect);
  c.set_access_channels(websocketpp::log::alevel::disconnect);

  c.init_asio();
  sherpa::Microphone mic;

  c.set_open_handler(
      [&c, &mic](connection_hdl hdl) { mic.StartMicrophone(&c, hdl); });

  c.set_message_handler(
      [&c](connection_hdl hdl, message_ptr msg) { OnMessage(&c, hdl, msg); });

  websocketpp::lib::error_code ec;
  client::connection_ptr con = c.get_connection(uri.str(), ec);
  if (ec) {
    std::cerr << "Could not create connection to " << uri.str()
              << " because: " << ec.message() << "\n";
    exit(EXIT_FAILURE);
  }
  c.connect(con);

  c.run();  // will exit when the above connection is closed

  SHERPA_LOG(INFO) << "Done!";
  return 0;
}
