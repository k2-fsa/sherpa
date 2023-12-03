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

#ifndef SHERPA_CPP_API_GRPC_ONLINE_GRPC_CLIENT_IMPL_H_
#define SHERPA_CPP_API_GRPC_ONLINE_GRPC_CLIENT_IMPL_H_

#include <iostream>
#include <memory>
#include <string>
#include <thread>  // NOLINT

#include "grpc/grpc.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"

#include "sherpa/csrc/log.h"
#include "sherpa/cpp_api/grpc/sherpa.grpc.pb.h"

namespace sherpa {

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;

class GrpcClient {
 public:
  GrpcClient(const std::string& host,
             int32_t port,
             int32_t nbest,
             const std::string& reqid);

  void SendBinaryData(const void* data, size_t size);
  void SetKey(std::string key) {key_ = key;}
  void Join();
  bool Done() const { return done_; }


 private:
  void ReadLoopFunc();
  void Connect();
  std::string host_;
  int32_t port_;
  int32_t nbest_;
  std::string reqid_;
  std::string key_;
  bool done_ = false;

  std::shared_ptr<Channel> channel_;
  std::unique_ptr<ASR::Stub> stub_;
  std::unique_ptr<ClientContext> context_;
  std::unique_ptr<ClientReaderWriter<Request, Response>> stream_;
  std::unique_ptr<Request> request_;
  std::unique_ptr<Response> response_;
  std::unique_ptr<std::thread> t_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_GRPC_ONLINE_GRPC_CLIENT_IMPL_H_
