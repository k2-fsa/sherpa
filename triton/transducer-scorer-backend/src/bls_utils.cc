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

#include "bls_utils.h"

namespace triton { namespace backend { namespace bls {

TRITONSERVER_Error*
CPUAllocator(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // For simplicity, this backend example always uses CPU memory regardless of
  // the preferred memory type. You can make the actual memory type and id that
  // we allocate be the same as preferred memory type. You can also provide a
  // customized allocator to support different preferred_memory_type, and reuse
  // memory buffer when possible.
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE, ("allocated " + std::to_string(byte_size) +
                                   " bytes for result tensor " + tensor_name)
                                      .c_str());
  } else {
    void* allocated_ptr = nullptr;
    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
    allocated_ptr = malloc(byte_size);

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          ("allocated " + std::to_string(byte_size) + " bytes in " +
           TRITONSERVER_MemoryTypeString(*actual_memory_type) +
           " for result tensor " + tensor_name)
              .c_str());
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }

  std::stringstream ss;
  ss << buffer;
  std::string buffer_str = ss.str();

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      ("Releasing buffer " + buffer_str + " of size " +
       std::to_string(byte_size) + " in " +
       TRITONSERVER_MemoryTypeString(memory_type) + " for result '" + *name)
          .c_str());

  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
    default:
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          std::string(
              "error: unexpected buffer allocated in CUDA managed memory")
              .c_str());
      break;
  }

  delete name;

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if (request != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "Failed to delete inference request.");
  }
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  // The following logic only works for non-decoupled models as for decoupled
  // models it may send multiple responses for a request or not send any
  // responses for a request. Need to modify this function if the model is using
  // decoupled API.
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

ModelExecutor::ModelExecutor(TRITONSERVER_Server* server) : server_(server)
{
  // When triton needs a buffer to hold an output tensor, it will ask
  // us to provide the buffer. In this way we can have any buffer
  // management and sharing strategy that we want. To communicate to
  // triton the functions that we want it to call to perform the
  // allocations, we create a "response allocator" object. We pass
  // this response allocate object to triton when requesting
  // inference. We can reuse this response allocator object for any
  // number of inference requests.
  allocator_ = nullptr;
  THROW_IF_TRITON_ERROR(TRITONSERVER_ResponseAllocatorNew(
      &allocator_, CPUAllocator, ResponseRelease, nullptr /* start_fn */));
}

TRITONSERVER_Error*
ModelExecutor::AsyncExecute(
    TRITONSERVER_InferenceRequest* irequest,
    std::future<TRITONSERVER_InferenceResponse*>* future)
{
  // Perform inference by calling TRITONSERVER_ServerInferAsync. This
  // call is asychronous and therefore returns immediately. The
  // completion of the inference and delivery of the response is done
  // by triton by calling the "response complete" callback functions
  // (InferResponseComplete in this case).
  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  *future = p->get_future();

  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
      irequest, allocator_, nullptr /* response_allocator_userp */,
      InferResponseComplete, reinterpret_cast<void*>(p)));

  RETURN_IF_ERROR(
      TRITONSERVER_ServerInferAsync(server_, irequest, nullptr /* trace */));

  return nullptr;  // success
}

}}}  // namespace triton::backend::bls
