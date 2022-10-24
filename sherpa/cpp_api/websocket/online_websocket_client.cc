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

#include "kaldi_native_io/csrc/kaldi-io.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/parse_options.h"
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

./bin/online_websocket_client --help

./bin/online_websocket_client \
  --server-ip=127.0.0.1 \
  --server-port=6006 \
  /path/to/foo.wav

)";

/** Read wave samples from a file.
 *
 * If the file has multiple channels, only the first channel is returned.
 * Samples are normalized to the range [-1, 1).
 *
 * @param filename Path to the wave file. Only "*.wav" format is supported.
 * @param expected_sample_rate  Expected sample rate of the wave file. It aborts
 *                              if the sample rate of the given file is not
 *                              equal to this value.
 *
 * @return Return a 1-D torch.float32 tensor containing audio samples
 * in the range [-1, 1)
 */
static torch::Tensor ReadWave(const std::string &filename,
                              float expected_sample_rate) {
  bool binary = true;
  kaldiio::Input ki(filename, &binary);
  kaldiio::WaveHolder wh;
  if (!wh.Read(ki.Stream())) {
    std::cerr << "Failed to read " << filename;
    exit(EXIT_FAILURE);
  }

  auto &wave_data = wh.Value();
  if (wave_data.SampFreq() != expected_sample_rate) {
    std::cerr << filename << "is expected to have sample rate "
              << expected_sample_rate << ". Given " << wave_data.SampFreq();
    exit(EXIT_FAILURE);
  }

  auto &d = wave_data.Data();

  if (d.NumRows() > 1) {
    std::cerr << "Only the first channel from " << filename << " is used";
  }

  auto tensor = torch::from_blob(const_cast<float *>(d.RowData(0)),
                                 {d.NumCols()}, torch::kFloat);

  return tensor / 32768;
}

static void OnMessage(client *c, const std::string &wave_filename,
                      connection_hdl hdl, message_ptr msg) {
  SHERPA_LOG(INFO) << "Decoding results for \n"
                   << wave_filename << "\n"
                   << msg->get_payload();
}

static void OnOpen(client *c, const std::string &filename, connection_hdl hdl) {
  auto samples = ReadWave(filename, 16000);
  int32_t num_samples = samples.numel();
  int32_t num_bytes = num_samples * sizeof(float);

  SHERPA_LOG(INFO) << "Decoding: " << filename;
  SHERPA_LOG(INFO) << "num_samples: " << num_samples;
  SHERPA_LOG(INFO) << "num_bytes: " << num_bytes;

  websocketpp::lib::error_code ec;
  c->send(hdl, samples.data_ptr<float>(), num_bytes,
          websocketpp::frame::opcode::binary, ec);
  if (ec) {
    std::cerr << "Failed to send audio samples";
    exit(EXIT_FAILURE);
  }

  return;

  ec.clear();
  c->send(hdl, "Done", websocketpp::frame::opcode::text, ec);

  if (ec) {
    std::cerr << "Failed to send Done\n";
    exit(EXIT_FAILURE);
  }

  sleep(2);

  ec.clear();
  c->close(hdl, websocketpp::close::status::normal, "I'm exiting now", ec);
  if (ec) {
    std::cerr << "Failed to close\n";
    exit(EXIT_FAILURE);
  }
}

int32_t main(int32_t argc, char *argv[]) {
  std::string server_ip = "127.0.0.1";
  int32_t server_port = 6006;

  sherpa::ParseOptions po(kUsageMessage);

  po.Register("server-ip", &server_ip, "IP address of the websocket server");
  po.Register("server-port", &server_port, "Port of the websocket server");

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

  std::string wave_filename = po.GetArg(1);

  bool secure = false;
  std::string resource = "/";
  websocketpp::uri uri(secure, server_ip, server_port, resource);

  client c;

  c.clear_access_channels(websocketpp::log::alevel::all);
  c.set_access_channels(websocketpp::log::alevel::connect);
  c.set_access_channels(websocketpp::log::alevel::disconnect);

  c.init_asio();

  c.set_open_handler([&c, &wave_filename](connection_hdl hdl) {
    OnOpen(&c, wave_filename, hdl);
  });

  c.set_message_handler(
      [&c, &wave_filename](connection_hdl hdl, message_ptr msg) {
        OnMessage(&c, wave_filename, hdl, msg);
      });

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
