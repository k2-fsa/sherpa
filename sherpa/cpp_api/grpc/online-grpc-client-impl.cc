// Copyright (c) 2021 Ximalaya Speech Team (Xiang Lyu)
//               2023 y00281951
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sherpa/cpp_api/grpc/online-grpc-client-impl.h"
#include "sherpa/csrc/log.h"

namespace sherpa {
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::Status;

GrpcClient::GrpcClient(const std::string& host,
                       int32_t port,
                       int32_t nbest,
                       const std::string& reqid)
    : host_(host),
      port_(port),
      nbest_(nbest),
      reqid_(reqid) {
  Connect();
  t_ = std::make_unique<std::thread>(&GrpcClient::ReadLoopFunc, this);
}

void GrpcClient::Connect() {
  channel_ = grpc::CreateChannel(host_ + ":" + std::to_string(port_),
                                 grpc::InsecureChannelCredentials());
  stub_ = ASR::NewStub(channel_);
  context_ = std::make_unique<ClientContext>();
  stream_ = stub_->Recognize(context_.get());
  request_ = std::make_unique<Request>();
  response_ = std::make_unique<Response>();
  request_->mutable_decode_config()->set_nbest_config(nbest_);
  request_->mutable_decode_config()->set_reqid(reqid_);
  stream_->Write(*request_);
}

void GrpcClient::SendBinaryData(const void* data, size_t size) {
  const int16_t* pdata = reinterpret_cast<const int16_t*>(data);
  request_->set_audio_data(pdata, size);
  stream_->Write(*request_);
}

void GrpcClient::ReadLoopFunc() {
  try {
    while (stream_->Read(response_.get())) {
      for (int32_t i = 0; i < response_->nbest_size(); i++) {
        // you can also traverse wordpieces like demonstrated above
        SHERPA_LOG(INFO) << i + 1 << "best " << response_->nbest(i).sentence();
      }
      if (response_->status() != Response_Status_ok) {
        break;
      }
      if (response_->type() == Response_Type_speech_end) {
        done_ = true;
        break;
      }
    }
  } catch (std::exception const& e) {
    SHERPA_LOG(ERROR) << e.what();
  }
}

void GrpcClient::Join() {
  stream_->WritesDone();
  t_->join();
  Status status = stream_->Finish();
  if (!status.ok()) {
    SHERPA_LOG(INFO) << "Recognize rpc failed.";
  }
}
}  // namespace sherpa

