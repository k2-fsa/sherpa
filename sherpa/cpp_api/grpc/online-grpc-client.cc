// sherpa/cpp_api/grpc/online-grpc-client.cc
//
// Copyright (c) 2023 y00281951

#include <chrono>  // NOLINT
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "grpc/grpc.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"

#include "kaldi_native_io/csrc/kaldi-table.h"
#include "kaldi_native_io/csrc/text-utils.h"
#include "kaldi_native_io/csrc/wave-reader.h"

#include "sherpa/csrc/fbank-features.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/file-utils.h"
#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/cpp_api/grpc/online-grpc-client-impl.h"

#define EXPECTED_SAMPLE_RATE 16000

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa using grpc.

Usage:

sherpa-online-grpc-client --help

sherpa-online-grpc-client \
  --server-ip=127.0.0.1 \
  --server-port=6006 \
  path/to/foo.wav \
  path/to/bar.wav \

or

sherpa-online-grpc-client \
  --server-ip=127.0.0.1 \
  --server-port=6006 \
  --use-wav-scp=true \
  scp:wav.scp \
  ark,scp,t:results.ark,results.scp
)";

void RecoginzeWav(std::string server_ip, int32_t server_port,
                  std::string req_id, std::string key,
                  const kaldiio::Matrix<float> &wav_data,
                  const float interval) {
  int32_t nbest = 1;
  const int32_t num_samples = wav_data.NumCols();
  const int32_t sample_interval = interval * EXPECTED_SAMPLE_RATE;

  sherpa::GrpcClient client(server_ip, server_port, nbest, req_id);
  client.SetKey(key);

  for (int32_t start = 0; start < num_samples; start += sample_interval) {
    if (client.Done()) {
      break;
    }
    int32_t end = std::min(start + sample_interval, num_samples);
    // Convert to short
    std::vector<int16_t> data;
    data.reserve(end - start);
    for (int32_t j = start; j < end; j++) {
      data.push_back(static_cast<int16_t>(wav_data(0, j)));
    }
    // Send PCM data
    client.SendBinaryData(data.data(), data.size() * sizeof(int16_t));
    SHERPA_LOG(INFO) << req_id << "Send " << data.size() << " samples";
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int32_t>(interval * 1000)));
  }
  client.Join();
}

int32_t main(int32_t argc, char* argv[]) {
  std::string server_ip = "127.0.0.1";
  int32_t server_port = 6006;
  bool use_wav_scp = false;    // true to use wav.scp as input

  sherpa::ParseOptions po(kUsageMessage);

  po.Register("server-ip", &server_ip, "IP address of the grpc server");
  po.Register("server-port", &server_port, "Port of the grpc server");
  po.Register("use-wav-scp", &use_wav_scp,
              "If true, user should provide two arguments: "
              "scp:wav.scp ark,scp,t:results.ark,results.scp");

  po.Read(argc, argv);

  if (server_port <= 0 || server_port > 65535) {
    SHERPA_LOG(FATAL) << "Invalid server port: " << server_port;
  }

  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  const float interval = 0.02;

  if (use_wav_scp) {
    SHERPA_CHECK_EQ(po.NumArgs(), 2)
        << "Please use something like:\n"
        << "scp:wav.scp ark,scp,t:results.scp,results.ark\n"
        << "if you provide --use-wav-scp=true";

    if (kaldiio::ClassifyRspecifier(po.GetArg(1), nullptr, nullptr) ==
        kaldiio::kNoRspecifier) {
      SHERPA_LOG(FATAL) << "Please provide an rspecifier. Current value is: "
                        << po.GetArg(1);
    }

    if (kaldiio::ClassifyWspecifier(po.GetArg(2), nullptr, nullptr, nullptr) ==
        kaldiio::kNoWspecifier) {
      SHERPA_LOG(FATAL) << "Please provide a wspecifier. Current value is: "
                        << po.GetArg(2);
    }

    kaldiio::TableWriter<kaldiio::TokenVectorHolder> writer(po.GetArg(2));

    kaldiio::SequentialTableReader<kaldiio::WaveHolder> wav_reader(
        po.GetArg(1));

    int32_t num_decoded = 0;
    for (; !wav_reader.Done(); wav_reader.Next()) {
      const std::string request_id = std::to_string(gen());

      SHERPA_LOG(INFO) << "\n" << num_decoded++ << ": decoding "
                       << wav_reader.Key();
      const auto &wave_data = wav_reader.Value();
      if (wave_data.SampFreq() != EXPECTED_SAMPLE_RATE) {
        SHERPA_LOG(FATAL) << wav_reader.Key()
                          << "is expected to have sample rate "
                          << EXPECTED_SAMPLE_RATE << ". Given "
                          << wave_data.SampFreq();
      }
      auto &d = wave_data.Data();
      if (d.NumRows() > 1) {
        SHERPA_LOG(WARNING)
            << "Only the first channel from " << wav_reader.Key() << " is used";
      }
      RecoginzeWav(server_ip, server_port, request_id,
                   wav_reader.Key(), d, interval);
    }
  } else {
    for (int32_t i = 1; i <= po.NumArgs(); ++i) {
      const std::string request_id = std::to_string(gen());
      bool binary = true;
      kaldiio::Input ki(po.GetArg(i), &binary);
      kaldiio::WaveHolder wh;
      if (!wh.Read(ki.Stream())) {
        SHERPA_LOG(FATAL) << "Failed to read " << po.GetArg(i);
      }
      auto &wave_data = wh.Value();
      if (wave_data.SampFreq() != EXPECTED_SAMPLE_RATE) {
        SHERPA_LOG(FATAL) << po.GetArg(i)
                          << "is expected to have sample rate "
                          << EXPECTED_SAMPLE_RATE << ". Given "
                          << wave_data.SampFreq();
      }
      auto &d = wave_data.Data();
      if (d.NumRows() > 1) {
        SHERPA_LOG(WARNING)
            << "Only the first channel from " << po.GetArg(i) << " is used";
      }
      RecoginzeWav(server_ip, server_port, request_id,
                   po.GetArg(i), d, interval);
    }
  }
  return 0;
}
