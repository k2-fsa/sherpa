// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <future>
#include <sstream>

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace triton {
namespace backend {
namespace scorer {

#define THROW_IF_TRITON_ERROR(X)                                       \
  do {                                                                 \
    TRITONSERVER_Error* tie_err__ = (X);                               \
    if (tie_err__ != nullptr) {                                        \
      throw BLSBackendException(TRITONSERVER_ErrorMessage(tie_err__)); \
    }                                                                  \
  } while (false)

//
// BLSBackendException
//
// Exception thrown if error occurs in BLSBackend.
//
struct BLSBackendException : std::exception {
  BLSBackendException(const std::string& message) : message_(message) {}

  const char* what() const throw() { return message_.c_str(); }

  std::string message_;
};

TRITONSERVER_Error* ResponseAlloc(TRITONSERVER_ResponseAllocator* allocator,
                                  const char* tensor_name, size_t byte_size,
                                  TRITONSERVER_MemoryType preferred_memory_type,
                                  int64_t preferred_memory_type_id, void* userp,
                                  void** buffer, void** buffer_userp,
                                  TRITONSERVER_MemoryType* actual_memory_type,
                                  int64_t* actual_memory_type_id);

// Callback functions for server inference.
TRITONSERVER_Error* ResponseRelease(TRITONSERVER_ResponseAllocator* allocator,
                                    void* buffer, void* buffer_userp,
                                    size_t byte_size,
                                    TRITONSERVER_MemoryType memory_type,
                                    int64_t memory_type_id);
void InferRequestComplete(TRITONSERVER_InferenceRequest* request,
                          const uint32_t flags, void* userp);
void InferResponseComplete(TRITONSERVER_InferenceResponse* response,
                           const uint32_t flags, void* userp);

//
// ModelExecutor
//
// Execute inference request on a model.
//
class ModelExecutor {
 public:
  ModelExecutor(TRITONSERVER_Server* server);

  // Performs async inference request.
  TRITONSERVER_Error* AsyncExecute(
      TRITONSERVER_InferenceRequest* irequest,
      std::future<TRITONSERVER_InferenceResponse*>* future);

 private:
  // The server object that encapsulates all the functionality of the Triton
  // server and allows access to the Triton server API.
  TRITONSERVER_Server* server_;

  // The allocator object that will be used for allocating output tensors.
  TRITONSERVER_ResponseAllocator* allocator_;
};

}  // namespace scorer
}  // namespace backend
}  // namespace triton
