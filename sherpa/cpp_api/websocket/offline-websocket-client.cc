// sherpa/cpp_api/websocket/offline-websocket-client.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include <string>

#include "kaldi_native_io/csrc/kaldi-io.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "sherpa/cpp_api/parse-options.h"
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

./bin/sherpa-offline-websocket-client --help

./bin/sherpa-offline-websocket-client \
  --server-ip=127.0.0.1 \
  --server-port=6006 \
  /path/to/foo.wav
)";

// Sample rate of the input wave. No resampling is made.
static constexpr int32_t kSampleRate = 16000;

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

class Client {
 public:
  Client(asio::io_context &io,  // NOLINT
         const std::string &ip, int16_t port, const std::string &wave_filename,
         float num_seconds_per_message)
      : io_(io),
        uri_(/*secure*/ false, ip, port, /*resource*/ "/"),
        samples_(ReadWave(wave_filename, kSampleRate)),
        samples_per_message_(num_seconds_per_message * kSampleRate) {
    c_.clear_access_channels(websocketpp::log::alevel::all);
    c_.set_access_channels(websocketpp::log::alevel::connect);
    c_.set_access_channels(websocketpp::log::alevel::disconnect);

    c_.init_asio(&io_);

    c_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });

    c_.set_close_handler(
        [](connection_hdl /*hdl*/) { SHERPA_LOG(INFO) << "Disconnected"; });

    c_.set_message_handler(
        [this](connection_hdl hdl, message_ptr msg) { OnMessage(hdl, msg); });

    Run();
  }

 private:
  void Run() {
    websocketpp::lib::error_code ec;
    client::connection_ptr con = c_.get_connection(uri_.str(), ec);
    if (ec) {
      SHERPA_LOG(ERROR) << "Could not create connection to " << uri_.str()
                        << " because: " << ec.message() << "\n";
      exit(EXIT_FAILURE);
    }

    c_.connect(con);
  }

  void OnOpen(connection_hdl hdl) {
    int32_t num_samples = samples_.numel();
    int32_t num_bytes = num_samples * sizeof(float);

    SHERPA_LOG(INFO) << "Sending " << num_bytes << " bytes\n";
    websocketpp::lib::error_code ec;
    c_.send(hdl, &num_bytes, sizeof(int32_t),
            websocketpp::frame::opcode::binary, ec);
    if (ec) {
      SHERPA_LOG(ERROR) << "Failed to send number of bytes because: "
                        << ec.message();
      exit(EXIT_FAILURE);
    }

    asio::post(io_, [this, hdl]() { this->SendMessage(hdl); });
  }

  void OnMessage(connection_hdl hdl, message_ptr msg) {
    SHERPA_LOG(INFO) << "Decoding results:\n" << msg->get_payload();

    websocketpp::lib::error_code ec;
    c_.send(hdl, "Done", websocketpp::frame::opcode::text, ec);

    if (ec) {
      SHERPA_LOG(ERROR) << "Failed to send Done because " << ec.message();
      exit(EXIT_FAILURE);
    }

    ec.clear();
    c_.close(hdl, websocketpp::close::status::normal, "I'm exiting now", ec);
    if (ec) {
      SHERPA_LOG(ERROR) << "Failed to close because " << ec.message();
      exit(EXIT_FAILURE);
    }
  }

  void SendMessage(connection_hdl hdl) {
    int32_t num_samples = samples_.numel();
    int32_t num_messages = num_samples / samples_per_message_;

    websocketpp::lib::error_code ec;

    if (num_sent_messages_ < num_messages) {
      SHERPA_LOG(INFO) << "Sending " << num_sent_messages_ << "/"
                       << num_messages << "\n";
      c_.send(hdl,
              samples_.data_ptr<float>() +
                  num_sent_messages_ * samples_per_message_,
              samples_per_message_ * sizeof(float),
              websocketpp::frame::opcode::binary, ec);

      if (ec) {
        SHERPA_LOG(INFO) << "Failed to send audio samples because "
                         << ec.message();
        exit(EXIT_FAILURE);
      }
      ec.clear();

      ++num_sent_messages_;
    }

    if (num_sent_messages_ == num_messages) {
      int32_t remaining_samples = num_samples % samples_per_message_;
      if (remaining_samples) {
        c_.send(hdl,
                samples_.data_ptr<float>() +
                    num_sent_messages_ * samples_per_message_,
                remaining_samples * sizeof(float),
                websocketpp::frame::opcode::binary, ec);

        if (ec) {
          SHERPA_LOG(INFO) << "Failed to send audio samples because "
                           << ec.message();
          exit(EXIT_FAILURE);
        }
      }
    } else {
      asio::post(io_, [this, hdl]() { this->SendMessage(hdl); });
    }
  }

 private:
  client c_;
  asio::io_context &io_;
  websocketpp::uri uri_;
  torch::Tensor samples_;

  int32_t samples_per_message_;
  int32_t num_sent_messages_ = 0;
};

int32_t main(int32_t argc, char *argv[]) {
  std::string server_ip = "127.0.0.1";
  int32_t server_port = 6006;
  float num_seconds_per_message = 10;

  sherpa::ParseOptions po(kUsageMessage);

  po.Register("server-ip", &server_ip, "IP address of the websocket server");
  po.Register("server-port", &server_port, "Port of the websocket server");
  po.Register("num-seconds-per-message", &num_seconds_per_message,
              "The number of samples per message equals to "
              "num_seconds_per_message*sample_rate");

  po.Read(argc, argv);
  SHERPA_CHECK_GT(num_seconds_per_message, 0);

  SHERPA_CHECK_GT(static_cast<int32_t>(num_seconds_per_message * kSampleRate),
                  0)
      << "num_seconds_per_message: " << num_seconds_per_message
      << ", kSampleRate: " << kSampleRate;

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

  asio::io_context io_conn;  // for network connections

  Client c(io_conn, server_ip, server_port, wave_filename,
           num_seconds_per_message);

  io_conn.run();  // will exit when the above connection is closed

  SHERPA_LOG(INFO) << "Done!";
  return 0;
}
