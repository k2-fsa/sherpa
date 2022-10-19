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
#include <mutex>  // NOLINT
#include <set>
#include <thread>  // NOLINT

#include "asio.hpp"
#include "sherpa/cpp_api/websocket/http_server.h"
#include "sherpa/cpp_api/websocket/tee_stream.h"
#include "websocketpp/config/asio_no_tls.hpp"  // TODO(fangjun): support TLS
#include "websocketpp/server.hpp"

using server = websocketpp::server<websocketpp::config::asio>;
using connection_hdl = websocketpp::connection_hdl;

class WebsocketServer {
 public:
  WebsocketServer(asio::io_context &io,  // NOLINT
                  const std::string &doc_root, const std::string &log_file)
      : io_(io),
        http_server_(doc_root),
        log_(log_file, std::ios::app),
        tee_(std::cout, log_) {
    server_.clear_access_channels(websocketpp::log::alevel::all);
    server_.set_access_channels(websocketpp::log::alevel::connect);
    server_.set_access_channels(websocketpp::log::alevel::disconnect);
    server_.set_access_channels(websocketpp::log::alevel::app);

    server_.get_alog().set_ostream(&tee_);
    server_.get_elog().set_ostream(&tee_);

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
    server_.set_reuse_addr(true);
    server_.listen(asio::ip::tcp::v4(), port);
    server_.start_accept();
  }

 private:
  void OnOpen(connection_hdl hdl) {
    auto c = server_.get_con_from_hdl(hdl);
    std::ostringstream os;
    os << std::this_thread::get_id()
       << " Connected: " << c->get_remote_endpoint() << "\n";
    server_.get_alog().write(websocketpp::log::alevel::app, os.str());

    std::lock_guard<std::mutex> lock(conn_mutex_);
    connections_.insert(hdl);
  }

  void OnClose(connection_hdl hdl) {
    auto c = server_.get_con_from_hdl(hdl);
    std::ostringstream os;
    os << std::this_thread::get_id()
       << " Disconnected: " << c->get_remote_endpoint() << "\n";
    server_.get_alog().write(websocketpp::log::alevel::app, os.str());

    std::lock_guard<std::mutex> lock(conn_mutex_);
    connections_.erase(hdl);
  }

  void OnHttp(connection_hdl hdl) {
    auto con = server_.get_con_from_hdl(hdl);

    std::string filename = con->get_resource();
    if (filename == "/") filename = "/index.html";

    std::string content;
    bool ret = http_server_.ProcessRequest(filename, &content);
    if (ret) {
      con->set_body(std::move(content));
      con->set_status(websocketpp::http::status_code::ok);
    } else {
      content = http_server_.GetErrorContent();
      con->set_body(std::move(content));
      con->set_status(websocketpp::http::status_code::not_found);
    }
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
        // TODO(fangjun):
        break;
    }
  }

 private:
  asio::io_context &io_;
  server server_;
  sherpa::HttpServer http_server_;

  // TODO(fangjun): Change it to a map, where the key is connection_hdl
  // and the value is OnlineStream
  std::set<connection_hdl, std::owner_less<connection_hdl>> connections_;
  std::mutex conn_mutex_;

  std::ofstream log_;
  sherpa::TeeStream tee_;
};

int main() {
  std::string doc_root = "./web";  // root for the http server

  // the log file is appended
  std::string log_file = "./log.txt";

  uint16_t port = 6006;
  int32_t num_threads = 1;  // thread pool size
  asio::io_context io;

  WebsocketServer srv(io, doc_root, log_file);
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
