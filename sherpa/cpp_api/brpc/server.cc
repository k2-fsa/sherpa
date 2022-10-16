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
#include <chrono>  //NOLINT
#include <condition_variable>  //NOLINT
#include <mutex>  //NOLINT
#include <random>
#include <shared_mutex>
#include <thread>  //NOLINT
#include <vector>

#include "asr.pb.h" //NOLINT
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_asr.h"
#include "sherpa/csrc/parse_options.h"

#include "brpc/server.h"
#include "brpc/stream.h"
#include "butil/base64.h"
#include "butil/logging.h"
#include "gflags/gflags.h"

using namespace std::chrono_literals;  //NOLINT

const int32_t MAX_ASR_RESULT = 512;
const int32_t SAMPLE_RATE = 16000;
const float PADDING_LEN = 0.36;
enum StreamState : int32_t {
  unknow_state = -1,
  input_start,
  vad_start,
  vad_continue,
  vad_end,
  vad_reset,
  input_end,
  input_reset
};

// AsrDecoder
class AsrDecoder final {
 public:
  AsrDecoder(const sherpa::OnlineAsrOptions opts,
      const size_t max_batch_size = 4096,
      const int32_t wait_ms = 10)
    : alive_(true),
    decoding_(false),
    wait_ms_(wait_ms),
    max_batch_size_(max_batch_size),
    online_asr_(opts) {
      batch_size_ = std::max(1,
          std::min(static_cast<int32_t>(max_batch_size_ >> 2), 64));
      decoder_ = std::thread(&AsrDecoder::batch_decode, this);
      decoder_.detach();
      srand(42);
    }

  ~AsrDecoder() {
    decoder_.join();
  }

  int64_t acquire_stream() {
    int64_t stream_id = -1;
    if (stream_queue_.size() <= max_batch_size_) {
      std::shared_lock mlock(mutex_);
      stream_id = rand() % max_batch_size_;  //NOLINT
      if (stream_queue_.find(stream_id) == stream_queue_.end()) {
        stream_queue_[stream_id] = online_asr_.CreateStream();
        result_queue_[stream_id] = std::vector<char>(MAX_ASR_RESULT);
        vad_state_queue_[stream_id] = input_start;
      }
    } else {
      SHERPA_LOG(WARNING) << "stream_queue_.size() and max_batch_size_ = "
        << max_batch_size_ << ", can't create more stream";
    }
    return stream_id;
  }

  void release_stream(const int64_t stream_id) {
    std::shared_lock mlock(mutex_);
    if (stream_queue_.find(stream_id) != stream_queue_.end()) {
      stream_queue_.erase(stream_id);
      result_queue_.erase(stream_id);
      vad_state_queue_.erase(stream_id);
    }
  }

  void reset_stream(const int64_t stream_id) {
    if (stream_queue_.find(stream_id) != stream_queue_.end()) {
      std::shared_lock mlock(mutex_);
      stream_queue_[stream_id] = online_asr_.CreateStream();
      result_queue_[stream_id] = std::vector<char>(MAX_ASR_RESULT);
      vad_state_queue_[stream_id] = input_start;
    }
  }

  void stop_stream(const int64_t stream_id) {
    if (stream_queue_.find(stream_id) != stream_queue_.end()) {
      vad_state_queue_[stream_id] = input_end;
    }
  }

  void push_stream(const int64_t stream_id,
      const char * data, const size_t len, bool end = false) {
    if (stream_queue_.find(stream_id) != stream_queue_.end()) {
      {
        std::shared_lock mlock(mutex_);
        if (vad_state_queue_[stream_id] == input_reset) {
          stream_queue_[stream_id] = online_asr_.CreateStream();
          vad_state_queue_[stream_id] = vad_reset;
        }
        const int num_samples = len / sizeof(int16_t);
        char * pcm_data = const_cast<char *>(data);
        auto wav_stream_tensor = torch::from_blob(
            reinterpret_cast<int16_t *>(pcm_data),
            {num_samples},
            torch::kInt16).to(torch::kFloat) / 32768;

        stream_queue_[stream_id]->AcceptWaveform(SAMPLE_RATE,
            wav_stream_tensor);
        if (end || vad_state_queue_[stream_id] == input_end) {
          stream_queue_[stream_id]->InputFinished();
          vad_state_queue_[stream_id] = input_reset;
        }
      }
      batch_cond_.notify_one();
    }
  }

  const char * fetch_stream(const int64_t stream_id) {
    if (stream_queue_.find(stream_id) != stream_queue_.end()) {
      std::shared_lock mlock(mutex_);
      decoding_cond_.wait_for(mlock, wait_ms_ * 1ms,
          [this] { return !decoding_; });
      std::string result =
        online_asr_.GetResult(stream_queue_[stream_id].get());
      result_queue_[stream_id].resize(MAX_ASR_RESULT);
      strncpy(result_queue_[stream_id].data(),
          result.c_str(),
          std::min(static_cast<int>(result.size()), MAX_ASR_RESULT));
      if (stream_queue_[stream_id]->IsEndpoint()) {
        vad_state_queue_[stream_id] = vad_end;
        stream_queue_[stream_id] = online_asr_.CreateStream();
        vad_state_queue_[stream_id] = vad_start;
      } else {
        vad_state_queue_[stream_id] = vad_continue;
      }
      return result_queue_[stream_id].data();
    }
    return nullptr;
  }

  int32_t segment_stream(const int64_t stream_id) {
    if (stream_queue_.find(stream_id) != stream_queue_.end()) {
      return vad_state_queue_[stream_id];
    }
    return unknow_state;
  }

  void batch_decode() {
    while (alive_) {
      {
        std::unique_lock mlock(mutex_);
        batch_cond_.wait_for(mlock, wait_ms_ * 1ms,
            [this] { return batch_size_ <= stream_queue_.size(); });

        std::vector<sherpa::OnlineStream *> ready_streams;
        for (const auto & stream : stream_queue_) {
          if (online_asr_.IsReady(stream.second.get())) {
            ready_streams.push_back(stream.second.get());
          }
        }

        if (ready_streams.empty()) { continue; }
        decoding_ = true;
        online_asr_.DecodeStreams(ready_streams.data(), ready_streams.size());
        decoding_ = false;
      }
      decoding_cond_.notify_all();
    }
  }

  void stop_decode() {
    std::unique_lock mlock(mutex_);
    alive_ = false;
  }

  void warmup() {
    //auto s = online_asr_.CreateStream();
    //s->AcceptWaveform(SAMPLE_RATE,
    //    wav_stream_tensor);
  }

 private:
  mutable std::shared_mutex mutex_;

  std::condition_variable_any batch_cond_;
  std::condition_variable_any decoding_cond_;

  bool alive_;
  bool decoding_;
  const int32_t wait_ms_;
  std::thread decoder_;

  size_t batch_size_;
  const size_t max_batch_size_;
  sherpa::OnlineAsr online_asr_;
  std::unordered_map<int64_t,
    std::unique_ptr<sherpa::OnlineStream>> stream_queue_;  //NOLINT
  std::unordered_map<int64_t, std::vector<char>> result_queue_;
  std::unordered_map<int64_t, int32_t> vad_state_queue_;
};

class StreamingAsrService : public sherpa::AsrService {
 public:
  explicit StreamingAsrService(const sherpa::OnlineAsrOptions opts,
      const int32_t max_batch_size = 4096,
      const int32_t wait_ms = 10)
    : decoder_(std::make_shared<AsrDecoder>(opts, max_batch_size, wait_ms)),
    tail_padding_(SAMPLE_RATE * sizeof(int16_t) * PADDING_LEN, 0) {}

  ~StreamingAsrService() {
    decoder_->stop_decode();
  }

  virtual void Recognize(google::protobuf::RpcController* controller,
      const sherpa::AsrRequest * request,
      sherpa::AsrResponse* response,
      google::protobuf::Closure* done) {
    try {
      sherpa::AsrResponse_Status status = sherpa::AsrResponse::ok;
      brpc::ClosureGuard done_guard(done);
      brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
      std::string remote_name(endpoint2str(cntl->remote_side()).c_str());
      if (remote_id_.find(remote_name) == remote_id_.end()) {
        std::unique_lock<std::mutex> mlock(mutex_);
        remote_id_[remote_name] = decoder_->acquire_stream();
      }

      // get PCM 16k16b1c data
      const char * pcm_data =
        reinterpret_cast<const char*>(request->audio_data().c_str());
      decoder_->push_stream(remote_id_[remote_name],
          pcm_data, request->audio_data().length());
      if (sherpa::AsrRequest::stream_end == request->status()) {
        decoder_->push_stream(remote_id_[remote_name],
            tail_padding_.data(), tail_padding_.size(), true);
      }
      const char * transcript = decoder_->fetch_stream(remote_id_[remote_name]);
      sherpa::AsrResponse_Status vad_state =
        decoder_->segment_stream(remote_id_[remote_name]) == vad_continue ?
        sherpa::AsrResponse::endpoint_inactive :
        sherpa::AsrResponse::endpoint_active;
      // update result
      if (transcript && strlen(transcript)) {
        LOG(INFO) << "remote " << cntl->remote_side()
          << " : pcm size=" << request->audio_data().length()
          << ", vad_status=" << vad_state
          << ", transcript = [" << transcript << "]";
        sherpa::AsrResponse_Path* one_path_ = response->add_nbest();
        one_path_->set_transcript(transcript);
        response->set_status(vad_state);
      }
      if (sherpa::AsrRequest::stream_end == request->status()) {
        decoder_->stop_stream(remote_id_[remote_name]);
      }
    } catch (std::exception const& e) {
      LOG(WARNING) << e.what();
    } catch (...) {
      LOG(WARNING) << "unknow error with StreamingAsrService";
    }
  }

 private:
  // batch decoder
  std::shared_ptr<AsrDecoder> decoder_;
  std::unordered_map<std::string, int64_t> remote_id_;
  std::vector<char> tail_padding_;

  std::mutex mutex_;
};

static constexpr const char *kUsageMessage = R"(
Online (streaming) automatic speech recognition RPC server with sherpa.

Usage:
(1) View help information.

  ./bin/brpc-server --help

(2) Run server

  ./bin/brpc-server \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --server-port=6006
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
  int32_t port = 6006;
  int32_t idle_timeout_s = 3;
  po.Register("server-port", &port, "Server port to listen on");
  po.Register("idle-timeout", &idle_timeout_s,
      "Connection will be closed in `idle_timeout_s'");
  po.Read(argc, argv);
  if (argc <= 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }
  opts.decoding_method = "modified_beam_search";
  // tips : trailing_silence for EndpointConfig is after sampling
  opts.endpoint_config.rule1 = sherpa::EndpointRule(false, 2.0, 0.0);
  opts.endpoint_config.rule3 = sherpa::EndpointRule(true, 1.2, 0.0);
  opts.endpoint_config.rule3 = sherpa::EndpointRule(false, 0.0, 20);
  opts.Validate();
  SHERPA_LOG(INFO) << "decoding method: " << opts.decoding_method;

  // create the server.
  brpc::Server server;
  StreamingAsrService asr_service_impl(opts);
  if (server.AddService(&asr_service_impl,
        brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Fail to add service";
    return -1;
  }
  brpc::ServerOptions options;
  options.idle_timeout_sec = idle_timeout_s;
  if (server.Start(port, &options) != 0) {
    LOG(ERROR) << "Fail to start AsrServer";
    return -1;
  }
  // wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server.RunUntilAskedToQuit();

  return 0;
}
