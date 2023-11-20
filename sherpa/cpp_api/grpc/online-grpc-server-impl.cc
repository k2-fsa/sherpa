// sherpa/cpp_api/grpc/online-grpc-server-impl.cc
//
// Copyright (c)  2022  Xiaomi Corporation
//                2023  y00281951

#include "sherpa/cpp_api/grpc/online-grpc-server-impl.h"
#include "sherpa/csrc/log.h"

#define SHERPA_SLEEP_TIME          100
#define SHERPA_SLEEP_ROUND_MAX     3000

namespace sherpa {
using grpc::ServerContext;
using grpc::ServerReaderWriter;

void OnlineGrpcDecoderConfig::Register(ParseOptions *po) {
  recognizer_config.Register(po);

  po->Register("loop-interval-ms", &loop_interval_ms,
               "It determines how often the decoder loop runs. ");

  po->Register("max-batch-size", &max_batch_size,
               "Max batch size for recognition.");
}

void OnlineGrpcDecoderConfig::Validate() const {
  recognizer_config.Validate();
  SHERPA_CHECK_GT(loop_interval_ms, 0);
  SHERPA_CHECK_GT(max_batch_size, 0);
}

void OnlineGrpcServerConfig::Register(sherpa::ParseOptions *po) {
  decoder_config.Register(po);
}

void OnlineGrpcServerConfig::Validate() const {
  decoder_config.Validate();
}

OnlineGrpcDecoder::OnlineGrpcDecoder(OnlineGrpcServer *server)
    : server_(server),
      config_(server->GetConfig().decoder_config),
      timer_(server->GetWorkContext()) {
  recognizer_ = std::make_unique<OnlineRecognizer>(config_.recognizer_config);
}

void OnlineGrpcDecoder::SerializeResult(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);
  auto result = recognizer_->GetResult(c->s.get());
  c->response_->clear_nbest();
  Response_OneBest* one_best_ = c->response_->add_nbest();
  one_best_->set_sentence(result.text);
}

void OnlineGrpcDecoder::OnPartialResult(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);
  if (!c->finish_flag_) {
    c->response_->set_status(Response::ok);
    c->response_->set_type(Response::partial_result);
    c->stream_->Write(*c->response_);
  }
}

void OnlineGrpcDecoder::OnFinalResult(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);
  if (!c->finish_flag_) {
    c->response_->set_status(Response::ok);
    c->response_->set_type(Response::final_result);
    c->stream_->Write(*c->response_);
  }
}

void OnlineGrpcDecoder::OnSpeechEnd(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);
  if (!c->finish_flag_) {
    c->response_->set_status(Response::ok);
    c->response_->set_type(Response::speech_end);
    c->stream_->Write(*c->response_);
  }
  c->finish_flag_ = true;
}

void OnlineGrpcDecoder::AcceptWaveform(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);
  float sample_rate =
      config_.recognizer_config.feat_config.fbank_opts.frame_opts.samp_freq;
  while (!c->samples.empty()) {
    c->s->AcceptWaveform(sample_rate, c->samples.front());
    c->samples.pop_front();
  }
}

void OnlineGrpcDecoder::InputFinished(std::shared_ptr<Connection> c) {
  std::lock_guard<std::mutex> lock(c->mutex);

  float sample_rate =
      config_.recognizer_config.feat_config.fbank_opts.frame_opts.samp_freq;

  while (!c->samples.empty()) {
    c->s->AcceptWaveform(sample_rate, c->samples.front());
    c->samples.pop_front();
  }

  // TODO(fangjun): Change the amount of paddings to be configurable
  torch::Tensor tail_padding =
      torch::zeros({static_cast<int64_t>(0.8 * sample_rate)}).to(torch::kFloat);

  c->s->AcceptWaveform(sample_rate, tail_padding);

  c->s->InputFinished();
}

void OnlineGrpcDecoder::Run() {
  timer_.expires_after(std::chrono::milliseconds(config_.loop_interval_ms));

  timer_.async_wait(
      [this](const asio::error_code &ec) { ProcessConnections(ec); });
}

void OnlineGrpcDecoder::ProcessConnections(const asio::error_code &ec) {
  if (ec) {
    SHERPA_LOG(FATAL) << "The decoder loop is aborted!";
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> to_remove;
  for (auto &p : connections_) {
    auto reqid = p.first;
    auto c = p.second;

    // The order of `if` below matters!
    if (!server_->Contains(reqid)) {
      // If the connection is disconnected, we stop processing it
      to_remove.push_back(reqid);
      continue;
    }

    if (active_.count(reqid)) {
      // Another thread is decoding this stream, so skip it
      continue;
    }

    if (!recognizer_->IsReady(c->s.get())) {
      // this stream has not enough frames to decode, so skip it
      continue;
    }

    // TODO(fangun): If the connection is timed out, we need to also
    // add it to `to_remove`

    // this stream has enough frames and is currently not processed by any
    // threads, so put it into the ready queue
    ready_connections_.push_back(c);

    // In `Decode()`, it will remove hdl from `active_`
    active_.insert(reqid);
  }

  for (auto reqid_rm : to_remove) {
    connections_.erase(reqid_rm);
  }

  if (!ready_connections_.empty()) {
    asio::post(server_->GetWorkContext(), [this]() { Decode(); });
  }

  // Schedule another call
  timer_.expires_after(std::chrono::milliseconds(config_.loop_interval_ms));

  timer_.async_wait(
      [this](const asio::error_code &ec) { ProcessConnections(ec); });
}

void OnlineGrpcDecoder::Decode() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (ready_connections_.empty()) {
    // There are no connections that are ready for decoding,
    // so we return directly
    return;
  }

  std::vector<std::shared_ptr<Connection>> c_vec;
  std::vector<OnlineStream *> s_vec;
  while (!ready_connections_.empty() &&
         static_cast<int32_t>(s_vec.size()) < config_.max_batch_size) {
    auto c = ready_connections_.front();
    ready_connections_.pop_front();

    c_vec.push_back(c);
    s_vec.push_back(c->s.get());
  }

  if (!ready_connections_.empty()) {
    // there are too many ready connections but this thread can only handle
    // max_batch_size connections at a time, so we schedule another call
    // to Decode() and let other threads to process the ready connections
    asio::post(server_->GetWorkContext(), [this]() { Decode(); });
  }

  lock.unlock();
  recognizer_->DecodeStreams(s_vec.data(), s_vec.size());
  lock.lock();

  for (auto c : c_vec) {
    auto result = recognizer_->GetResult(c->s.get());
    SerializeResult(c);
    if (!result.is_final) {
      OnPartialResult(c);
    } else {
      OnFinalResult(c);
      connections_.erase(c->reqid_);
      OnSpeechEnd(c);
    }
    SHERPA_LOG(INFO) << "Decode result:" << result.AsJsonString();
    active_.erase(c->reqid_);
  }
}

OnlineGrpcServer::OnlineGrpcServer(
    asio::io_context &io_work,
    const OnlineGrpcServerConfig &config)
    : config_(config),
      io_work_(io_work),
      decoder_(this) {}

void OnlineGrpcServer::Run() {
  decoder_.Run();
}

bool OnlineGrpcServer::Contains(std::string reqid) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return connections_.count(reqid);
}

Status OnlineGrpcServer::Recognize(ServerContext* context,
                             ServerReaderWriter<Response, Request>* stream) {
  SHERPA_LOG(INFO) << "Get Recognize request";
  std::shared_ptr<OnlineStream> s = decoder_.recognizer_->CreateStream();
  auto c = std::make_shared<Connection> (
              std::make_shared<ServerReaderWriter<Response, Request>>(*stream),
              std::make_shared<Request>(),
              std::make_shared<Response>(),
              s);
  int32_t sleep_cnt = 0;

  float sample_rate = decoder_.config_.recognizer_config.
                      feat_config.fbank_opts.frame_opts.samp_freq;

  while (stream->Read(c->request_.get())) {
    if (!c->start_flag_) {
      c->start_flag_ = true;
      c->reqid_ = c->request_->decode_config().reqid();

      mutex_.lock();
      connections_.insert(c->reqid_);
      mutex_.unlock();

      decoder_.mutex_.lock();
      decoder_.connections_.insert({c->reqid_, c});
      decoder_.mutex_.unlock();
    } else {
      const int16_t* pcm_data =
           reinterpret_cast<const int16_t*>(c->request_->audio_data().c_str());
      int32_t num_samples =
                          c->request_->audio_data().length() / sizeof(int16_t);
      SHERPA_LOG(INFO) << c->reqid_ << "Received "
                       << num_samples << " samples";
      torch::Tensor samples = torch::from_blob(const_cast<int16_t *>(pcm_data),
                                      {num_samples},
                                      torch::kShort).to(torch::kFloat) / 32768;
      samples = samples.clone();
      c->samples.push_back(samples);
      decoder_.AcceptWaveform(c);
    }
  }
  decoder_.InputFinished(c);

  while (!c->finish_flag_) {
    std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int32_t>(SHERPA_SLEEP_TIME)));
    if (sleep_cnt++ > SHERPA_SLEEP_ROUND_MAX) {
      c->finish_flag_ = true;
      break;
    }
  }

  mutex_.lock();
  connections_.erase(c->reqid_);
  mutex_.unlock();

  SHERPA_LOG(INFO) << "reqid:" << c->reqid_ << " Connection close";
  return Status::OK;
}
}  // namespace sherpa
