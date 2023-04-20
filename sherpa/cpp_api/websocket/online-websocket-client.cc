// sherpa/cpp_api/websocket/online-websocket-client.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include <chrono>  // NOLINT
#include <fstream>
#include <string>

#include "kaldi_native_io/csrc/kaldi-io.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "nlohmann/json.hpp"
#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/log.h"
#include "torch/script.h"
#include "websocketpp/client.hpp"
#include "websocketpp/config/asio_no_tls_client.hpp"
#include "websocketpp/uri.hpp"

using json = nlohmann::json;
using client = websocketpp::client<websocketpp::config::asio_client>;

using message_ptr = client::message_ptr;
using websocketpp::connection_hdl;

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa using websocket.

Usage:

./bin/sherpa-online-websocket-client --help

./bin/sherpa-online-websocket-client \
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
  std::cout << filename;
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
         float seconds_per_message, int32_t SampleRate,
         std::string ctm_filename)
      : io_(io),
        uri_(/*secure*/ false, ip, port, /*resource*/ "/"),
        samples_(ReadWave(wave_filename, SampleRate)),
        samples_per_message_(seconds_per_message * SampleRate),
        seconds_per_message_(seconds_per_message),
        ctm_filename_(ctm_filename) {
    c_.clear_access_channels(websocketpp::log::alevel::all);
    //    c_.set_access_channels(websocketpp::log::alevel::connect);
    //    c_.set_access_channels(websocketpp::log::alevel::disconnect);
    of_ = std::ofstream(ctm_filename);
    of_ << std::fixed << std::setprecision(2);
    std::string base_filename =
        wave_filename.substr(wave_filename.find_last_of("/\\") + 1);
    wave_filename_ = base_filename.substr(0, base_filename.find_last_of('.'));

    c_.init_asio(&io_);
    c_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });
    c_.set_close_handler(
        [this](connection_hdl /*hdl*/) { SHERPA_LOG(INFO) << "Disconnected"; });
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

  void DumpCtm(nlohmann::json result) {
    int i = 0;
    std::vector<std::string> tokens =
        result["tokens"].get<std::vector<std::string>>();
    int length = tokens.size();
    if (length < 1) {
      return;
    }
    std::vector<float> timestamps =
        result["timestamps"].get<std::vector<float>>();
    if (tokens[0].at(0) != ' ') {
      SHERPA_LOG(WARNING) << "First word is not a new word " << tokens[0];
    }

    std::string word = tokens[0];
    float start_time = result["start_time"];
    float start = timestamps[0] + start_time;
    float duration = 0.01;
    if (length > 2) {
      duration = timestamps[1] - timestamps[0];
    }
    int word_start_index = i;
    while (i < length) {
      //      SHERPA_LOG(INFO) <<i<<" "<<length<< tokens[i];
      while (i + 1 < length && tokens[i + 1].at(0) != ' ') {
        word += tokens[i + 1];
        if (length > i + 2) {
          duration = timestamps[i + 2] - timestamps[word_start_index];
        }
        i++;
      }
      if (word.compare(" ") != 0) {
        of_ << wave_filename_ << " 0 " << start << " " << duration << " "
            << word << std::endl;
      }
      if (i >= length - 1) {
        break;
      }
      i++;
      word_start_index = i;
      word = tokens[i];
      start = timestamps[i] + start_time;
      duration = 0.01;
      if (length > i + 1) {
        duration = timestamps[i + 1] - timestamps[word_start_index];
      }
    }
  }

  void OnOpen(connection_hdl hdl) {
    auto start_time = std::chrono::steady_clock::now();
    asio::post(
        io_, [this, hdl, start_time]() { this->SendMessage(hdl, start_time); });
  }

  void OnMessage(connection_hdl hdl, message_ptr msg) {
    const std::string &payload = msg->get_payload();
    auto result = json::parse(payload);
    std::string res = result.dump();
    SHERPA_LOG(INFO) << res;
    if (result["segment"] > segment_id_) {
      segment_id_ = result["segment"];
      std::cout << text_;
      if (ctm_filename_.length() > 0) {
        DumpCtm(old_result_);
      }
    }
    text_ = result["text"].get<std::string>();
    old_result_ = result;
    if (result["final"]) {
      std::cout << result["text"].get<std::string>() << std::endl;
      if (ctm_filename_.length() > 0) {
        DumpCtm(result);
      }
      websocketpp::lib::error_code ec;
      c_.close(hdl, websocketpp::close::status::normal, "I'm exiting now", ec);
      if (ec) {
        SHERPA_LOG(INFO) << "Failed to close because " << ec.message();
        exit(EXIT_FAILURE);
      }
    }
  }

  void SendMessage(
      connection_hdl hdl,
      std::chrono::time_point<std::chrono::steady_clock> start_time) {
    int32_t num_samples = samples_.numel();
    int32_t num_messages = num_samples / samples_per_message_;

    websocketpp::lib::error_code ec;
    auto time = std::chrono::steady_clock::now();
    int elapsed_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time)
            .count();
    if (elapsed_time_ms <
        static_cast<int>(seconds_per_message_ * num_sent_messages_ * 1000)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(int(
          seconds_per_message_ * num_sent_messages_ * 1000 - elapsed_time_ms)));
    }
    if (num_sent_messages_ < 1) {
      SHERPA_LOG(INFO) << "Starting to send audio";
    }
    if (num_sent_messages_ < num_messages) {
      // SHERPA_LOG(DEBUG) << "Sending " << num_sent_messages_ << "/"
      //                  << num_messages << "\n";
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
        ec.clear();
      }
      c_.send(hdl, "Done", websocketpp::frame::opcode::text, ec);
      SHERPA_LOG(INFO) << "Sent Done Signal";
      if (ec) {
        SHERPA_LOG(INFO) << "Failed to send Done because " << ec.message();
        exit(EXIT_FAILURE);
      }
    } else {
      asio::post(io_, [this, hdl, start_time]() {
        this->SendMessage(hdl, start_time);
      });
    }
  }

 private:
  client c_;
  asio::io_context &io_;
  websocketpp::uri uri_;
  torch::Tensor samples_;
  nlohmann::json old_result_;
  int32_t samples_per_message_;
  int32_t num_sent_messages_ = 0;
  float seconds_per_message_;
  int32_t segment_id_ = 0;
  std::string text_;
  std::string wave_filename_;
  std::string ctm_filename_;
  std::ofstream of_;
};

int32_t main(int32_t argc, char *argv[]) {
  std::string server_ip = "127.0.0.1";
  int32_t server_port = 6006;
  float seconds_per_message = 10;
  // Sample rate of the input wave. No resampling is made.
  int32_t SampleRate = 16000;
  std::string ctm_filename = "";

  sherpa::ParseOptions po(kUsageMessage);

  po.Register("server-ip", &server_ip, "IP address of the websocket server");
  po.Register("server-port", &server_port, "Port of the websocket server");
  po.Register("samplerate", &SampleRate,
              "SampleRate of the recorded audio (expecting wav, no resampling "
              "is done)");
  po.Register("num-seconds-per-message", &seconds_per_message,
              "The number of samples per message equals to "
              "seconds_per_message*sample_rate");
  po.Register("ctm-filename", &ctm_filename, "Name of the CTM output file");

  po.Read(argc, argv);
  SHERPA_CHECK_GT(seconds_per_message, 0);
  SHERPA_CHECK_GT(static_cast<int32_t>(seconds_per_message * SampleRate), 0)
      << "seconds_per_message: " << seconds_per_message
      << ", SampleRate: " << SampleRate;

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
  Client c(io_conn, server_ip, server_port, wave_filename, seconds_per_message,
           SampleRate, ctm_filename);

  io_conn.run();  // will exit when the above connection is closed

  SHERPA_LOG(INFO) << "Done!";
  return 0;
}
