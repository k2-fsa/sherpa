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

#include <set>
#include <thread>

#include "asio.hpp"
#include "websocketpp/config/asio_no_tls.hpp"  // TODO(fangjun): support TLS
#include "websocketpp/server.hpp"

using server = websocketpp::server<websocketpp::config::asio>;
using connection_hdl = websocketpp::connection_hdl;

class Server {
 public:
  Server(asio::io_context &io) : io_(io) {
    server_.set_access_channels(websocketpp::log::alevel::all);
    server_.clear_access_channels(websocketpp::log::alevel::frame_payload);

    server_.init_asio(&io_);

    server_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });

    server_.set_close_handler([this](connection_hdl hdl) { OnClose(hdl); });

    server_.set_message_handler(
        [this](connection_hdl hdl, server::message_ptr msg) {
          OnMessage(hdl, msg);
        });
  }

  void Run(uint16_t port) {
    server_.listen(asio::ip::tcp::v4(), port);
    websocketpp::lib::error_code ec;
    server_.start_accept(ec);
  }

 private:
  void OnOpen(connection_hdl hdl) {
    auto c = server_.get_con_from_hdl(hdl);
    std::cout << "Connected: " << c->get_remote_endpoint() << "\n";

    connections_.insert(hdl);
  }
  void OnClose(connection_hdl hdl) {
    auto c = server_.get_con_from_hdl(hdl);
    std::cout << "Disconnected: " << c->get_remote_endpoint() << "\n";
    connections_.erase(hdl);
  }

  void OnMessage(connection_hdl hdl, server::message_ptr msg) {
    switch (msg->get_opcode()) {
      case websocketpp::frame::opcode::text:
        // process text
        std::cout << "text: " << msg->get_payload() << "\n";
        break;
      case websocketpp::frame::opcode::binary:
        // process binary
        break;
      default:
        // TODO:
        break;
    }
  }

 private:
  asio::io_context &io_;
  server server_;

  std::set<connection_hdl, std::owner_less<connection_hdl>> connections_;
};

int main() {
  uint16_t port = 6006;
  int32_t num_threads = 1;  // thread pool size
  asio::io_context io;

  Server srv(io);
  srv.Run(port);
  std::cout << "Listening on: " << port << "\n";

  std::vector<std::thread> threads;
  for (int32_t i = 0; i != num_threads; ++i) {
    threads.emplace_back([&io]() { io.run(); });
  }

  io.run();

  for (auto &t : threads) {
    t.join();
  }

  return 0;
}
