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

#include "sherpa/cpp_api/websocket/online_websocket_server_impl.h"

#include "sherpa/csrc/file_utils.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

void OnlineWebsocketDecoderConfig::Register(ParseOptions *po) {
  po->Register("nn-model", &nn_model, "Path to the torchscript model");

  po->Register("tokens", &tokens, "Path to tokens.txt");

  po->Register("decoding-method", &decoding_method,
               "Decoding method to use. Possible values are: greedy_search, "
               "modified_beam_search");

  po->Register("num-active-paths", &num_active_paths,
               "Number of active paths for modified_beam_search. "
               "Used only when --decoding-method is modified_beam_search");

  po->Register("use-gpu", &use_gpu,
               "True to use GPU for computation."
               "Caution: We currently assume there is only one GPU. You have "
               "to change the code to support multiple GPUs.");

  po->Register(
      "max-batch-size", &max_batch_size,
      "Max batch size for decoding. If you are using CPU, increasing "
      "it will increase the memory usage if there are many active "
      "connections. We suggest that you use a small value for it for "
      "CPU decoding, e.g., 5, since it is pretty fast for CPU decoding");

  po->Register(
      "max-utterance-length", &max_utterance_length,
      "Max utterance length in seconds. If we receive an utterance "
      "longer than this value, we will reject the connection. "
      "If you have enough memory, you can select a large value for it.");
}

void OnlineWebsocketDecoderConfig::Validate() const {
  if (nn_model.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --nn-model";
  }

  if (!FileExists(nn_model)) {
    SHERPA_LOG(FATAL) << "\n--nn-model=" << nn_model << "\n"
                      << nn_model << " does not exist!";
  }

  if (tokens.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --tokens";
  }

  if (!FileExists(tokens)) {
    SHERPA_LOG(FATAL) << "\n--tokens=" << tokens << "\n"
                      << tokens << " does not exist!";
  }

  if (decoding_method != "greedy_search" &&
      decoding_method != "modified_beam_search") {
    SHERPA_LOG(FATAL)
        << "Unsupported decoding method: " << decoding_method
        << ". Supported values are: greedy_search, modified_beam_search";
  }

  if (decoding_method == "modified_beam_search") {
    SHERPA_CHECK_GT(num_active_paths, 0);
  }

  SHERPA_CHECK_GT(max_batch_size, 0);

  SHERPA_CHECK_GT(max_utterance_length, 0);
}

void OnlineWebsocketServerConfig::Register(sherpa::ParseOptions *po) {}

void OnlineWebsocketServerConfig::Validate() const {}

OnlineWebsocketServer::OnlineWebsocketServer(
    asio::io_context &io_conn, asio::io_context &io_work,
    const OnlineWebsocketServerConfig &config,
    const OnlineWebsocketDecoderConfig &decoder_config)
    : io_conn_(io_conn),
      io_work_(io_work),
      log_(config.log_file, std::ios::app),
      tee_(std::cout, log_),
      decoder_config_(decoder_config) {
  SetupLog();

  server_.init_asio(&io_conn_);

  server_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });

  server_.set_close_handler([this](connection_hdl hdl) { OnClose(hdl); });

  // server_.set_http_handler([this](connection_hdl hdl) { OnHttp(hdl); });

  server_.set_message_handler(
      [this](connection_hdl hdl, server::message_ptr msg) {
        OnMessage(hdl, msg);
      });

  sherpa::DecodingOptions opts;
  if (decoder_config.decoding_method == "greedy_search") {
    opts.method = sherpa::kGreedySearch;
  } else if (decoder_config.decoding_method == "modified_beam_search") {
    opts.method = sherpa::kModifiedBeamSearch;
    opts.num_active_paths = decoder_config.num_active_paths;
  }
  recognizer_ = std::make_unique<OnlineRecognizer>(
      decoder_config.nn_model, decoder_config.tokens, opts,
      decoder_config.use_gpu, decoder_config.sample_rate);
}

void OnlineWebsocketServer::Run(uint16_t port) {
  server_.set_reuse_addr(true);
  server_.listen(asio::ip::tcp::v4(), port);
  server_.start_accept();
}

void OnlineWebsocketServer::SetupLog() {
  server_.clear_access_channels(websocketpp::log::alevel::all);
  server_.set_access_channels(websocketpp::log::alevel::connect);
  server_.set_access_channels(websocketpp::log::alevel::disconnect);

  // So that it also prints to std::cout and std::cerr
  server_.get_alog().set_ostream(&tee_);
  server_.get_elog().set_ostream(&tee_);
}

void OnlineWebsocketServer::OnOpen(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  SHERPA_LOG(INFO) << "New connection: "
                   << server_.get_con_from_hdl(hdl)->get_remote_endpoint();

  connections_.emplace(hdl, recognizer_->CreateStream());

  SHERPA_LOG(INFO) << "Number of active connections: " << connections_.size()
                   << "\n";
}
void OnlineWebsocketServer::OnClose(connection_hdl hdl) {
  // std::lock_guard<std::mutex> lock(mutex_);
  // connections_.erase(hdl);

  SHERPA_LOG(INFO) << "Number of active connections: " << connections_.size()
                   << "\n";
}

void OnlineWebsocketServer::OnMessage(connection_hdl hdl,
                                      server::message_ptr msg) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto stream = connections_.find(hdl)->second;
  lock.unlock();
  const std::string &payload = msg->get_payload();

  switch (msg->get_opcode()) {
    case websocketpp::frame::opcode::text:
      if (payload == "Done") {
        SHERPA_LOG(INFO) << "received done\n";
        // stream->InputFinished();
        // TODO(fangjun): Send it to decode
      }
      break;
    case websocketpp::frame::opcode::binary: {
      auto p = reinterpret_cast<const float *>(payload.data());
      int32_t num_samples = payload.size() / sizeof(float);
      SHERPA_LOG(INFO) << "num_samples: " << num_samples;
      torch::Tensor samples = torch::from_blob(const_cast<float *>(p),
                                               {num_samples}, torch::kFloat);
      stream->AcceptWaveform(decoder_config_.sample_rate, samples);
      while (recognizer_->IsReady(stream.get())) {
        // TODO(fangjun): Send it to a queue for decoding
        recognizer_->DecodeStream(stream.get());
      }
      auto result = recognizer_->GetResult(stream.get());

      websocketpp::lib::error_code ec;
      server_.send(hdl, result, websocketpp::frame::opcode::text, ec);
      if (ec) {
        server_.get_alog().write(websocketpp::log::alevel::app, ec.message());
      }
      SHERPA_LOG(INFO) << "result:" << result;

      SHERPA_LOG(INFO) << "done\n";
      break;
    }
    default:
      break;
  }
}

void OnlineWebsocketServer::Close(connection_hdl hdl,
                                  websocketpp::close::status::value code,
                                  const std::string &reason) {
  auto con = server_.get_con_from_hdl(hdl);

  std::ostringstream os;
  os << "Closing " << con->get_remote_endpoint() << " with reason: " << reason
     << "\n";

  websocketpp::lib::error_code ec;
  server_.close(hdl, code, reason, ec);
  if (ec) {
    os << "Failed to close" << con->get_remote_endpoint() << ". "
       << ec.message() << "\n";
  }
  server_.get_alog().write(websocketpp::log::alevel::app, os.str());
}

}  // namespace sherpa
