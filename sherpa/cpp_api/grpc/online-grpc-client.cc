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

#include "sherpa/csrc/wav.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/file-utils.h"
#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/cpp_api/grpc/online-grpc-client-impl.h"

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa using grpc.

Usage:

sherpa-online-grpc-client --help

sherpa-online-grpc-client \
  --server-ip=127.0.0.1 \
  --server-port=6006 \
  --wav-path=/path/to/foo.wav

or

sherpa-online-grpc-client \
  --server-ip=127.0.0.1 \
  --server-port=6006 \
  --wav-scp=/path/to/wav.scp
)";

void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

int32_t main(int32_t argc, char* argv[]) {
  std::string server_ip = "127.0.0.1";
  int32_t server_port = 6006;
  std::string wav_path;
  std::string wav_scp;
  sherpa::ParseOptions po(kUsageMessage);

  po.Register("server-ip", &server_ip, "IP address of the grpc server");
  po.Register("server-port", &server_port, "Port of the grpc server");
  po.Register("wav-path", &wav_path, "wav to recognize");
  po.Register("wav-scp", &wav_scp, "wav.scp path");

  po.Read(argc, argv);

  if (server_port <= 0 || server_port > 65535) {
    SHERPA_LOG(FATAL) << "Invalid server port: " << server_port;
  }

  if (po.NumArgs() != 0 || (wav_scp == "" && wav_path == "")) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  const int32_t sample_rate = 16000;
  const float interval = 0.02;
  const int32_t sample_interval = interval * sample_rate;

  std::vector<std::pair<std::string, std::string>> wav_dict;
  if (wav_path != "") {
    wav_dict.push_back({wav_path, wav_path});
  } else if (wav_scp != "" && sherpa::FileExists(wav_scp)) {
    std::ifstream in(wav_scp);
    std::string line;
    while (getline(in, line)) {
      std::string wav_id;
      std::string wav_path;
      std::vector<std::string> res;
      SplitStringToVector(line, "\t", true, &res);
      if (res.size() != 2) {
        SplitStringToVector(line, " ", true, &res);
      }
      if (res.size() != 2) {
        SHERPA_LOG(WARNING) << "Wav scp: " << wav_scp << " format error";
        continue;
      }
      wav_id = res[0];
      wav_path = res[1];
      if (!sherpa::FileExists(wav_path)) {
        SHERPA_LOG(WARNING) << "Wav path: " << wav_path << " not exist";
        continue;
      }
      wav_dict.push_back(make_pair(wav_id, wav_path));
    }
  }

  if (wav_dict.size() == 0) {
    SHERPA_LOG(WARNING) << "There is no wav to decode, please check!";
    return -1;
  }

  for (size_t i = 0; i < wav_dict.size(); i++) {
    int32_t req_id = gen();
    int32_t nbest = 1;
    const std::string request_id = std::to_string(req_id);
    sherpa::GrpcClient client(server_ip, server_port, nbest, request_id);
    client.key_ = wav_dict[i].first;
    sherpa::WavReader wav_reader(wav_dict[i].second);
    const int32_t num_samples = wav_reader.num_samples();
    std::vector<float> pcm_data(wav_reader.data(),
                              wav_reader.data() + num_samples);

    for (int32_t start = 0; start < num_samples; start += sample_interval) {
      if (client.Done()) {
        break;
      }
      int32_t end = std::min(start + sample_interval, num_samples);
      // Convert to short
      std::vector<int16_t> data;
      data.reserve(end - start);
      for (int32_t j = start; j < end; j++) {
        data.push_back(static_cast<int16_t>(pcm_data[j]));
      }
      // Send PCM data
      client.SendBinaryData(data.data(), data.size() * sizeof(int16_t));
      SHERPA_LOG(INFO) << req_id << "Send " << data.size() << " samples";
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int32_t>(interval * 1000)));
    }
    client.Join();
  }
  return 0;
}
