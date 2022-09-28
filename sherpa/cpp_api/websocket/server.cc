// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "sherpa/cpp_api/websocket/server.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_asr.h"
#include "sherpa/csrc/parse_options.h"


static constexpr const char *kUsageMessage = R"(
Online (streaming) automatic speech recognition RPC server with sherpa.

Usage:
(1) View help information.

  ./bin/websocket-server --help

(2) Run server

  ./bin/websocket-server \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --port=6006
)";

int main(int argc, char *argv[]) {
  // set torch
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;
  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  // set OnlineAsr option
  sherpa::ParseOptions po(kUsageMessage);
  sherpa::OnlineAsrOptions opts;
  opts.Register(&po);
  int port;
  po.Register("server-port", &port, "Server port to listen on");
  po.Read(argc, argv);
  SHERPA_LOG(INFO) << "decoding method: " << opts.decoding_method;
  opts.Validate();

  sherpa::WebSocketServer server(port, opts);
  SHERPA_LOG(INFO) << "Listening at port " << port;
  server.Start();
  return 0;
}
