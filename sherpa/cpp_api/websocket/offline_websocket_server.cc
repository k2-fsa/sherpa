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

#include <thread>  // NOLINT

#include "asio.hpp"
#include "sherpa/cpp_api/websocket/offline_websocket_server_impl.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/parse_options.h"
#include "torch/all.h"

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa using websocket.

Usage:

./bin/offline_websocket_server --help

./bin/offline_websocket_server \
  --use-gpu=false \
  --port=6006 \
  --num-io-threads=3 \
  --num-work-threads=5 \
  --max-batch-size=5 \
  --nn-model=/path/to/cpu.jit \
  --tokens=/path/to/tokens.txt \
  --decoding-method=greedy_search \
  --doc-root=../sherpa/bin/web \
  --log-file=./log.txt
)";

int32_t main(int32_t argc, char *argv[]) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  sherpa::ParseOptions po(kUsageMessage);

  sherpa::OfflineWebsocketServerConfig config;
  sherpa::OfflineWebsocketDecoderConfig decoder_config;

  // the server will listen on this port, for both websocket and http
  int32_t port = 6006;

  // size of the thread pool for handling network connections
  int32_t num_io_threads = 3;

  // size of the thread pool for neural network computation and decoding
  int32_t num_work_threads = 5;

  po.Register("num-io-threads", &num_io_threads,
              "Number of threads to use for network connections.");

  po.Register("num-work-threads", &num_work_threads,
              "Number of threads to use for neural network "
              "computation and decoding.");

  po.Register("port", &port, "The port on which the server will listen.");

  config.Register(&po);
  decoder_config.Register(&po);

  po.Read(argc, argv);

  config.Validate();
  decoder_config.Validate();

  asio::io_context io_conn;  // for network connections
  asio::io_context io_work;  // for neural network and decoding

  sherpa::OfflineWebsocketServer server(io_conn, io_work, config,
                                        decoder_config);
  server.Run(port);

  SHERPA_LOG(INFO) << "Listening on: " << port << "\n";
  SHERPA_LOG(INFO) << "Number of I/O threads: " << num_io_threads << "\n";
  SHERPA_LOG(INFO) << "Number of work threads: " << num_work_threads << "\n";

  // give some work to do for the io_work pool
  auto work_guard = asio::make_work_guard(io_work);

  std::vector<std::thread> io_threads;

  // decrement since the main thread is also used for network communications
  for (int32_t i = 0; i < num_io_threads - 1; ++i) {
    io_threads.emplace_back([&io_conn]() { io_conn.run(); });
  }

  std::vector<std::thread> work_threads;
  for (int32_t i = 0; i < num_work_threads; ++i) {
    work_threads.emplace_back([&io_work]() { io_work.run(); });
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
