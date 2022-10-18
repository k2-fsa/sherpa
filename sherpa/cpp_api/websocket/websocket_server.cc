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

#include <fstream>
#include <mutex>
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

    server_.set_http_handler([this](connection_hdl hdl) { OnHttp(hdl); });
  }

  void Run(uint16_t port) {
    server_.listen(asio::ip::tcp::v4(), port);
    websocketpp::lib::error_code ec;
    server_.start_accept(ec);
  }

 private:
  void OnOpen(connection_hdl hdl) {
    auto c = server_.get_con_from_hdl(hdl);
    std::cout << std::this_thread::get_id()
              << " Connected: " << c->get_remote_endpoint() << "\n";

    std::lock_guard<std::mutex> lock(conn_mutex_);
    connections_.insert(hdl);
  }

  void OnClose(connection_hdl hdl) {
    auto c = server_.get_con_from_hdl(hdl);
    std::cout << std::this_thread::get_id()
              << " Disconnected: " << c->get_remote_endpoint() << "\n";

    std::lock_guard<std::mutex> lock(conn_mutex_);
    connections_.erase(hdl);
  }

  // TODO(fangjun): Write a separate class to process HTTP requests.
  // Also, pre-load files into memory.
  void OnHttp(connection_hdl hdl) {
    auto con = server_.get_con_from_hdl(hdl);
    std::cout << std::this_thread::get_id()
              << " Http Connected: " << con->get_remote_endpoint() << "\n";

    std::string filename = con->get_resource();
    std::cout << std::this_thread::get_id() << " filename: " << filename
              << "\n";

    std::ostringstream os;
    if (filename == "/") {
      filename = "web/index.html";
    } else {
      os << "<!doctype html><html><head>"
         << "<title>Speech recognition with next-gen Kaldi</title><body>"
         << "<h1>Hello world 404</h1>"
         << "</body></head></html>";

      con->set_body(os.str());
      con->set_status(websocketpp::http::status_code::not_found);
      return;
    }

    std::ifstream file;
    std::string response;
    file.open(filename.c_str(), std::ios::in);
    file.seekg(0, std::ios::end);
    response.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    response.assign((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
    con->set_body(std::move(response));
    con->set_status(websocketpp::http::status_code::ok);
  }

  void OnMessage(connection_hdl hdl, server::message_ptr msg) {
    switch (msg->get_opcode()) {
      case websocketpp::frame::opcode::text:
        // process text
        std::cout << std::this_thread::get_id()
                  << " text: " << msg->get_payload() << "\n";
        break;
      case websocketpp::frame::opcode::binary:
        // process binary
        std::cout << std::this_thread::get_id()
                  << " binary: " << msg->get_payload() << "\n";
        break;
      default:
        // TODO:
        break;
    }
  }

 private:
  asio::io_context &io_;
  server server_;

  // TODO(fangjun): Change it to a map, where the key is connection_hdl
  // and the value is OnlineStream
  std::set<connection_hdl, std::owner_less<connection_hdl>> connections_;
  std::mutex conn_mutex_;
};

int main() {
  uint16_t port = 6006;
  int32_t num_threads = 1;  // thread pool size
  asio::io_context io;

  Server srv(io);
  srv.Run(port);
  std::cout << std::this_thread::get_id() << " Listening on: " << port << "\n";

  std::vector<std::thread> threads;
  for (int32_t i = 0; i != num_threads; ++i) {
    threads.emplace_back([&io]() {
      std::cout << std::this_thread::get_id() << " thread started\n";
      io.run();
    });
  }

  io.run();

  for (auto &t : threads) {
    t.join();
  }

  return 0;
}
