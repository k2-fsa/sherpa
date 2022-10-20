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
#include "sherpa/cpp_api/offline_recognizer.h"
#include "sherpa/cpp_api/websocket/http_server.h"
#include "sherpa/cpp_api/websocket/tee_stream.h"
#include "sherpa/csrc/parse_options.h"
#include "torch/all.h"
#include "websocketpp/config/asio_no_tls.hpp"  // TODO(fangjun): support TLS
#include "websocketpp/server.hpp"

using server = websocketpp::server<websocketpp::config::asio>;
using connection_hdl = websocketpp::connection_hdl;

struct ConnectionData {
  int32_t expected_byte_size = 0;
  int32_t cur = 0;
  std::vector<int8_t> data;

  void Clear() {
    expected_byte_size = 0;
    cur = 0;
    data.clear();
  }
};
struct WebsocketServerConfig {
  std::string nn_model = "./cpu_jit.pt";
  std::string tokens = "./tokens.txt";

  std::string doc_root = "./web";  // root for the http server

  // the log file is appended
  std::string log_file = "./log.txt";

  float sample_rate = 16000;
  bool use_gpu = false;
  int32_t max_batch_size = 10;

  void Register(sherpa::ParseOptions *po) {
    po->Register("nn-model", &nn_model, "Path to the torchscript model");
    po->Register("tokens", &tokens, "Path to tokens.txt");
    po->Register("doc-root", &doc_root,
                 "Path to the directory where "
                 "files like index.html for the HTTP server locate");
    po->Register("log-file", &log_file, "Path to the log file");
    po->Register("use-gpu", &use_gpu, "True to use GPU for computation");
    po->Register("max-batch-size", &max_batch_size, "max batch size");
  }
};

class WebsocketServer {
 public:
  WebsocketServer(asio::io_context &io_conn,  // NOLINT
                  asio::io_context &io_work,  // NOLINT
                  const WebsocketServerConfig &config)
      : io_conn_(io_conn),
        io_work_(io_work),
        http_server_(config.doc_root),
        log_(config.log_file, std::ios::app),
        tee_(std::cout, log_),
        config_(config) {
    server_.clear_access_channels(websocketpp::log::alevel::all);
    server_.set_access_channels(websocketpp::log::alevel::connect);
    server_.set_access_channels(websocketpp::log::alevel::disconnect);
    server_.set_access_channels(websocketpp::log::alevel::app);

    server_.get_alog().set_ostream(&tee_);
    server_.get_elog().set_ostream(&tee_);

    server_.init_asio(&io_conn_);

    server_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });

    server_.set_close_handler([this](connection_hdl hdl) { OnClose(hdl); });

    server_.set_message_handler(
        [this](connection_hdl hdl, server::message_ptr msg) {
          OnMessage(hdl, msg);
        });

    server_.set_http_handler([this](connection_hdl hdl) { OnHttp(hdl); });

    // init asr

    sherpa::DecodingOptions opts;
    opts.method = sherpa::kGreedySearch;
    std::cout << "init asr\n";
    offline_recognizer_ = std::make_unique<sherpa::OfflineRecognizer>(
        config.nn_model, config.tokens, opts, config.use_gpu,
        config.sample_rate);
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
    auto it = connections_.find(hdl);
    if (it == connections_.end()) {
      // this is a new connection
      connections_.emplace(hdl, ConnectionData{});
    }
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
    auto c = server_.get_con_from_hdl(hdl);
    websocketpp::lib::error_code ec;

    std::unique_lock<std::mutex> lock(conn_mutex_);
    auto it = connections_.find(hdl);
    lock.unlock();

    auto &connection_data = it->second;

    std::ostringstream os;
    switch (msg->get_opcode()) {
      case websocketpp::frame::opcode::text: {
        const auto &payload = msg->get_payload();
        if (payload == "DONE") {
          // The client will not send any more data. We can close the
          // connection now.
          Close(hdl, websocketpp::close::status::normal, "Done");
        } else {
          Close(hdl, websocketpp::close::status::normal,
                std::string("Invalid payload: ") + payload);
        }
        break;
      }

      case websocketpp::frame::opcode::binary: {
        const std::string &payload = msg->get_payload();
        const int8_t *p = reinterpret_cast<const int8_t *>(payload.data());
        if (connection_data.expected_byte_size == 0) {
          // the first packet (assume the current machine is little endian)
          connection_data.expected_byte_size =
              *reinterpret_cast<const int32_t *>(p);

          connection_data.data.resize(connection_data.expected_byte_size);
          std::copy(payload.begin() + 4, payload.end(),
                    connection_data.data.data());
          connection_data.cur = payload.size() - 4;
        } else {
          std::copy(payload.begin(), payload.end(),
                    connection_data.data.data() + connection_data.cur);
          connection_data.cur += payload.size();
        }

        if (connection_data.expected_byte_size == connection_data.cur) {
          {
            std::unique_lock<std::mutex> lock(streams_mutex_);
            streams_.push_back(hdl);
          }

          asio::post(io_work_, [this]() { Decode(); });
        }
        break;
      }
      default:
        break;
    }
    server_.get_alog().write(websocketpp::log::alevel::app, os.str());
  }

 private:
  void Close(connection_hdl hdl, websocketpp::close::status::value code,
             const std::string &reason) {
    websocketpp::lib::error_code ec;
    server_.close(hdl, code, reason, ec);
    if (ec) {
      std::ostringstream os;
      os << "Failed to close"
         << server_.get_con_from_hdl(hdl)->get_remote_endpoint() << "."
         << "Reason was: " << reason << "\n";
      server_.get_alog().write(websocketpp::log::alevel::app, os.str());
    }
  }

  void Decode() {
    std::unique_lock<std::mutex> lock_stream(streams_mutex_);
    if (streams_.empty()) {
      return;
    }

    int32_t size = std::min<int32_t>(streams_.size(), config_.max_batch_size);
    std::vector<connection_hdl> handles(size);
    for (int32_t i = 0; i != size; ++i) {
      connection_hdl hdl = streams_.front();
      handles[i] = hdl;
      streams_.pop_front();
    }

    lock_stream.unlock();

    std::ostringstream os;
    os << "batch size: " << size << "\n";
    server_.get_alog().write(websocketpp::log::alevel::app, os.str());

    std::vector<const float *> samples(size);
    std::vector<int32_t> samples_length(size);

    std::unique_lock<std::mutex> lock_connection(conn_mutex_);
    for (int32_t i = 0; i != size; ++i) {
      std::cout << "take " << i << "\n";
      const auto &connection_data = connections_.at(handles[i]);
      auto f = reinterpret_cast<const float *>(
          const_cast<int8_t *>(&connection_data.data[0]));
      int32_t num_samples = connection_data.expected_byte_size / sizeof(float);
      samples[i] = (f);
      samples_length[i] = num_samples;
    }
    lock_connection.unlock();

    auto results = offline_recognizer_->DecodeSamplesBatch(
        samples.data(), samples_length.data(), size);

    for (int32_t i = 0; i != size; ++i) {
      connection_hdl hdl = handles[i];
      asio::post(io_conn_, [this, hdl, text = results[i].text]() {
        server_.send(hdl, text, websocketpp::frame::opcode::text);
      });
    }

    lock_connection.lock();
    for (int32_t i = 0; i != size; ++i) {
      connections_[handles[i]].Clear();
    }
  }

 private:
  asio::io_context &io_conn_;
  asio::io_context &io_work_;
  server server_;
  sherpa::HttpServer http_server_;

  std::map<connection_hdl, ConnectionData, std::owner_less<connection_hdl>>
      connections_;

  std::mutex conn_mutex_;

  std::ofstream log_;
  sherpa::TeeStream tee_;
  std::unique_ptr<sherpa::OfflineRecognizer> offline_recognizer_;

  std::deque<connection_hdl> streams_;
  std::mutex streams_mutex_;
  WebsocketServerConfig config_;
};

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa.

Usage:

./bin/websocketpp_server --help

./bin/websocketpp_server \

)";

int32_t main(int32_t argc, char *argv[]) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  sherpa::ParseOptions po(kUsageMessage);

  WebsocketServerConfig config;

  int32_t port = 6006;

  // size of the thread pool for handling network connections
  int32_t num_io_threads = 1;

  // size of the thread pool for neural network computation and decoding
  int32_t num_work_threads = 2;

  po.Register("num-io-threads", &num_io_threads,
              "Number of threads to use for network connections.");

  po.Register("num-work-threads", &num_work_threads,
              "Number of threads to use for neural network "
              "computation and decoding.");

  po.Register("port", &port, "The port on which the server will listen.");

  config.Register(&po);

  po.Read(argc, argv);

  asio::io_context io_conn;  // for network connections
  asio::io_context io_work;  // for neural network and decoding

  WebsocketServer srv(io_conn, io_work, config);
  srv.Run(port);
  std::cout << std::this_thread::get_id() << " Listening on: " << port << "\n";
  std::cout << "Number of I/O threads: " << num_io_threads << "\n";
  std::cout << "Number of work threads: " << num_work_threads << "\n";

  // give some work to do for the io_work pool
  auto work_guard = asio::make_work_guard(io_work);

  std::vector<std::thread> io_threads;

  // decrement since the main thread is also used for network communications
  for (int32_t i = 0; i < num_io_threads - 1; ++i) {
    io_threads.emplace_back([&io_conn]() {
      std::cout << std::this_thread::get_id() << " I/O thread started\n";
      io_conn.run();
    });
  }

  std::vector<std::thread> work_threads;
  for (int32_t i = 0; i < num_work_threads; ++i) {
    work_threads.emplace_back([&io_work]() {
      std::cout << std::this_thread::get_id() << " work thread started\n";
      io_work.run();
    });
  }

  io_conn.run();

  for (auto &t : io_threads) {
    t.join();
  }

  for (auto &t : work_threads) {
    t.join();
  }

  return 0;
}
