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

#include "sherpa/cpp_api/websocket/offline_websocket_server_impl.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa/csrc/file_utils.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

void OfflineWebsocketDecoderConfig::Register(ParseOptions *po) {
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

void OfflineWebsocketDecoderConfig::Validate() const {
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

OfflineWebsocketDecoder::OfflineWebsocketDecoder(
    const OfflineWebsocketDecoderConfig &config, OfflineWebsocketServer *server)
    : config_(config), server_(server) {
  DecodingOptions opts;

  if (config.decoding_method == "greedy_search") {
    opts.method = kGreedySearch;
  } else if (config.decoding_method == "modified_beam_search") {
    opts.method = kModifiedBeamSearch;
    opts.num_active_paths = config.num_active_paths;
  } else {
    SHERPA_LOG(FATAL) << "Unsupported decoding method: "
                      << config.decoding_method;
  }

  offline_recognizer_ = std::make_unique<OfflineRecognizer>(
      config.nn_model, config.tokens, opts, config.use_gpu, config.sample_rate);
}

void OfflineWebsocketDecoder::Push(connection_hdl hdl, ConnectionDataPtr d) {
  std::lock_guard<std::mutex> lock(mutex_);
  streams_.push_back({hdl, d});
}

void OfflineWebsocketDecoder::Decode() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (streams_.empty()) {
    return;
  }

  int32_t size =
      std::min(static_cast<int32_t>(streams_.size()), config_.max_batch_size);

  // We first lock the mutex for streams_, take items from it, and then
  // unlock the mutex; in doing so we don't need to lock the mutex to
  // access hdl and connection_data later.
  std::vector<connection_hdl> handles(size);

  // Store connection_data here to prevent the data from being freed
  // while we are still using it.
  std::vector<ConnectionDataPtr> connection_data(size);

  std::vector<const float *> samples(size);
  std::vector<int32_t> samples_length(size);

  for (int32_t i = 0; i != size; ++i) {
    auto &p = streams_.front();
    handles[i] = p.first;
    connection_data[i] = p.second;
    streams_.pop_front();

    samples[i] = reinterpret_cast<const float *>(&connection_data[i]->data[0]);

    samples_length[i] = connection_data[i]->expected_byte_size / sizeof(float);
  }

  lock.unlock();

  // Note: DecodeSamplesBatch is thread-safe
  auto results = offline_recognizer_->DecodeSamplesBatch(
      samples.data(), samples_length.data(), size);

  for (int32_t i = 0; i != size; ++i) {
    connection_hdl hdl = handles[i];
    asio::post(server_->GetConnectionContext(),
               [this, hdl, text = results[i].text]() {
                 websocketpp::lib::error_code ec;
                 server_->GetServer().send(
                     hdl, text, websocketpp::frame::opcode::text, ec);
                 if (ec) {
                   server_->GetServer().get_alog().write(
                       websocketpp::log::alevel::app, ec.message());
                 }
               });
  }
}
void OfflineWebsocketServerConfig::Register(ParseOptions *po) {
  po->Register("doc-root", &doc_root,
               "Path to the directory where "
               "files like index.html for the HTTP server locate");

  po->Register("log-file", &log_file,
               "Path to the log file. Logs are "
               "appended to this file");
}

void OfflineWebsocketServerConfig::Validate() const {
  if (doc_root.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --doc-root, e.g., sherpa/bin/web";
  }

  if (!FileExists(doc_root + "/index.html")) {
    SHERPA_LOG(FATAL) << "\n--doc-root=" << doc_root << "\n"
                      << doc_root << "/index.html does not exist!";
  }
}

OfflineWebsocketServer::OfflineWebsocketServer(
    asio::io_context &io_conn,  // NOLINT
    asio::io_context &io_work,  // NOLINT
    const OfflineWebsocketServerConfig &config,
    const OfflineWebsocketDecoderConfig &decoder_config)
    : io_conn_(io_conn),
      io_work_(io_work),
      http_server_(config.doc_root),
      config_(config),
      log_(config.log_file, std::ios::app),
      tee_(std::cout, log_),
      decoder_(decoder_config, this) {
  SetupLog();

  server_.init_asio(&io_conn_);

  server_.set_open_handler([this](connection_hdl hdl) { OnOpen(hdl); });

  server_.set_close_handler([this](connection_hdl hdl) { OnClose(hdl); });

  server_.set_http_handler([this](connection_hdl hdl) { OnHttp(hdl); });

  server_.set_message_handler(
      [this](connection_hdl hdl, server::message_ptr msg) {
        OnMessage(hdl, msg);
      });

  max_byte_size_ =
      static_cast<int32_t>(decoder_config.max_utterance_length *
                           decoder_config.sample_rate * sizeof(float));

  SHERPA_LOG(INFO) << "max_utterance_length: "
                   << decoder_config.max_utterance_length << " s,"
                   << "max_byte_size_: " << max_byte_size_;
}

void OfflineWebsocketServer::SetupLog() {
  server_.clear_access_channels(websocketpp::log::alevel::all);
  server_.set_access_channels(websocketpp::log::alevel::connect);
  server_.set_access_channels(websocketpp::log::alevel::disconnect);

  // So that it also prints to std::cout and std::cerr
  server_.get_alog().set_ostream(&tee_);
  server_.get_elog().set_ostream(&tee_);
}

void OfflineWebsocketServer::OnOpen(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.emplace(hdl, std::make_shared<ConnectionData>());

  SHERPA_LOG(INFO) << "Number of active connections: " << connections_.size()
                   << "\n";
}

void OfflineWebsocketServer::OnClose(connection_hdl hdl) {
  std::lock_guard<std::mutex> lock(mutex_);
  connections_.erase(hdl);

  SHERPA_LOG(INFO) << "Number of active connections: " << connections_.size()
                   << "\n";
}

void OfflineWebsocketServer::OnHttp(connection_hdl hdl) {
  auto con = server_.get_con_from_hdl(hdl);

  std::string filename = con->get_resource();
  if (filename == "/") filename = "/index.html";

  std::string content;
  bool found = false;
  if (filename != "/streaming_record.html") {
    found = http_server_.ProcessRequest(filename, &content);
  } else {
    content = R"(
<!doctype html><html><head>
<title>Speech recognition with next-gen Kaldi</title><body>
<h2>/streaming_record.html is not available for the offline server"</h2>;
<br/>
<br/>
Go back to <a href="/upload.html">/upload.html</a> or
<a href="/offline_record.html">/offline_record.html</a>
</body></head></html>
    )";
  }

  if (found) {
    con->set_status(websocketpp::http::status_code::ok);
  } else {
    con->set_status(websocketpp::http::status_code::not_found);
  }

  con->set_body(std::move(content));
}

void OfflineWebsocketServer::OnMessage(connection_hdl hdl,
                                       server::message_ptr msg) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto connection_data = connections_.find(hdl)->second;
  lock.unlock();
  const std::string &payload = msg->get_payload();

  switch (msg->get_opcode()) {
    case websocketpp::frame::opcode::text:
      if (payload == "Done") {
        // The client will not send any more data. We can close the
        // connection now.
        Close(hdl, websocketpp::close::status::normal, "Done");
      } else {
        Close(hdl, websocketpp::close::status::normal,
              std::string("Invalid payload: ") + payload);
      }
      break;

    case websocketpp::frame::opcode::binary: {
      auto p = reinterpret_cast<const int8_t *>(payload.data());

      if (connection_data->expected_byte_size == 0) {
        if (payload.size() < 4) {
          Close(hdl, websocketpp::close::status::normal,
                "Payload is too short");
          break;
        }

        // the first packet (assume the current machine is little endian)
        connection_data->expected_byte_size =
            *reinterpret_cast<const int32_t *>(p);

        if (connection_data->expected_byte_size > max_byte_size_) {
          float num_samples =
              connection_data->expected_byte_size / sizeof(float);
          float duration = num_samples / decoder_.GetConfig().sample_rate;

          std::ostringstream os;
          os << "Max utterance length is configured to "
             << decoder_.GetConfig().max_utterance_length
             << " seconds, received length is " << duration << " seconds. "
             << "Payload is too large!";
          SHERPA_LOG(INFO) << os.str();
          Close(hdl, websocketpp::close::status::message_too_big, os.str());
          break;
        }

        connection_data->data.resize(connection_data->expected_byte_size);
        std::copy(payload.begin() + 4, payload.end(),
                  connection_data->data.data());
        connection_data->cur = payload.size() - 4;
      } else {
        std::copy(payload.begin(), payload.end(),
                  connection_data->data.data() + connection_data->cur);
        connection_data->cur += payload.size();
      }

      if (connection_data->expected_byte_size == connection_data->cur) {
        auto d = std::make_shared<ConnectionData>(std::move(*connection_data));
        decoder_.Push(hdl, d);

        connection_data->Clear();

        asio::post(io_work_, [this]() { decoder_.Decode(); });
      }
      break;
    }

    default:
      // Unexpected message, ignore it
      break;
  }
}

void OfflineWebsocketServer::Close(connection_hdl hdl,
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

void OfflineWebsocketServer::Run(uint16_t port) {
  server_.set_reuse_addr(true);
  server_.listen(asio::ip::tcp::v4(), port);
  server_.start_accept();
}

}  // namespace sherpa
