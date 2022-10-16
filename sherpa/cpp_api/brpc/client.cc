/**
 * Copyright      2022  (authors: Pingfeng Luo)
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
#include <stdio.h>

#include "asr.pb.h" //NOLINT
#include "brpc/channel.h"
#include "brpc/stream.h"
#include "bthread/bthread.h"
#include "butil/logging.h"
#include "gflags/gflags.h"


DEFINE_string(protocol, "baidu_std",
    "Protocol type, defined in src/brpc/options.proto");
DEFINE_string(connection_type, "",
    "Connection type in [single, pooled, short]");
DEFINE_string(server, "0.0.0.0:6006", "IP Address of server");
DEFINE_string(load_balancer, "", "The algorithm for load balancing");
DEFINE_int32(timeout_ms, 5000, "RPC timeout in milliseconds");
DEFINE_int32(connect_timeout_ms, 3000,
    "RPC connect timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");
DEFINE_string(wav_file, "", "Wav file to test");

void HandleAsrResponse(
    brpc::Controller* cntl,
    sherpa::AsrResponse* response) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<sherpa::AsrResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(WARNING) << "Fail to send AsrRequest, " << cntl->ErrorText();
    return;
  }
  if (response->nbest_size()) {
    LOG(INFO) << "Received response from " << cntl->remote_side()
      << ": status=[" << response->status()
      << "], transcript=[" << response->nbest(0).transcript()
      << "], latency=[" << cntl->latency_us() << "us" << "]";
  }
}

int main(int argc, char* argv[]) {
  GFLAGS_NS::ParseCommandLineFlags(&argc, &argv, true);

  // a Channel represents a communication line to a Server.
  brpc::Channel channel;

  brpc::ChannelOptions options;
  options.protocol = FLAGS_protocol;
  options.connect_timeout_ms = FLAGS_connect_timeout_ms;
  options.connection_type = FLAGS_connection_type;
  options.timeout_ms = FLAGS_timeout_ms/*milliseconds*/;
  options.max_retry = FLAGS_max_retry;
  if (channel.Init(FLAGS_server.c_str(),
        FLAGS_load_balancer.c_str(), &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel";
    return -1;
  }

  // normally, you should not call a Channel directly, but instead construct
  // a stub Service wrapping it. stub can be shared by all threads as well.
  sherpa::AsrService_Stub stub(&channel);

  // send a request and wait for the response every 0.32 second.
  const float sample_rate = 16000;
  const float frame_time = 0.32;  // second
  size_t frame_size = sample_rate * frame_time * sizeof(int16_t);
  std::vector<char> pcm(frame_size);
  FILE * fp = fopen(FLAGS_wav_file.c_str(), "rb");
  if (fp) {
    fseek(fp, 44, SEEK_SET);
  }

  while (fp &&
      (frame_size =
       fread(reinterpret_cast<char*>(pcm.data()), sizeof(char), frame_size, fp))
      >= 0) {
    // since we are sending asynchronous RPC (`done' is not NULL),
    // these objects MUST remain valid until `done' is called.
    // as a result, we allocate these objects on heap
    sherpa::AsrResponse* response = new sherpa::AsrResponse();
    brpc::Controller* cntl = new brpc::Controller();

    // notice that you don't have to new request, which can be modified
    // or destroyed just after stub.Recognize is called.
    // send data
    sherpa::AsrRequest request;
    request.set_audio_data(reinterpret_cast<const int16_t *>(pcm.data()),
        frame_size);

    sherpa::AsrRequest_Status status = sherpa::AsrRequest::stream_continue;
    if (frame_size == 0) {
      status = sherpa::AsrRequest::stream_end;
    }
    request.set_status(status);

    // in asynchronous RPC, we fetch the result inside the callback
    google::protobuf::Closure* done = brpc::NewCallback(
        &HandleAsrResponse, cntl, response);
    stub.Recognize(cntl, &request, response, done);
    size_t sleep_us = frame_time * 1000000;
    if (frame_size == 0) {
      sleep_us *= 10;
    }
    bthread_usleep(sleep_us);
    if (frame_size == 0) {
      break;
    }
  }
  while (!brpc::IsAskedToQuit()) {}

  LOG(INFO) << "AsrClient is going to quit";
  return 0;
}
