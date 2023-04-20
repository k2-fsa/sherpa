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

#include "bls.h"

namespace triton {
namespace backend {
namespace scorer {

BLSExecutor::BLSExecutor(TRITONSERVER_Server* server)
    : server_(server), model_executor_(server) {}

TRITONSERVER_Error* BLSExecutor::PrepareInferenceRequest(
    TRITONSERVER_InferenceRequest** irequest, const std::string& model_name) {
  // Create an inference request object. The inference request object
  // is where we set the name of the model we want to use for
  // inference and the input tensors.
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestNew(
      irequest, server_, model_name.c_str(), -1 /* model_version */));

  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
      *irequest, InferRequestComplete, nullptr /* request_release_userp */));

  return nullptr;  // success
}

TRITONSERVER_Error* BLSExecutor::PrepareInferenceInput(
    const std::vector<torch::Tensor>& input_tensors,
    const std::vector<const char*>& input_names,
    TRITONSERVER_InferenceRequest* irequest) {
  size_t input_count;
  input_count = input_tensors.size();

  const char* name;
  TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_FP16;

  TRITONSERVER_MemoryType data_memory_type = TRITONSERVER_MEMORY_GPU;
  int64_t data_memory_id = 0;  // TODO: get from config

  for (size_t count = 0; count < input_count; count++) {
    name = input_names[count];
    if (std::strcmp(name, "y") == 0) {
      // FIX ME, hard-code for decoder
      datatype = TRITONSERVER_TYPE_INT64;
    }
    std::vector<int64_t> input_shapes;
    auto shape = input_tensors[count].sizes();
    input_shapes.reserve(shape.size());
    for (auto itr = shape.begin(); itr != shape.end(); itr++) {
      input_shapes.push_back(*itr);
    }
    uint32_t dims_count = (uint32_t)input_shapes.size();

    const char* data_buffer =
        reinterpret_cast<const char*>(input_tensors[count].data_ptr());
    size_t data_byte_size = input_tensors[count].nbytes();

    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddInput(
        irequest, name, datatype, &input_shapes[0], dims_count));

    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
        irequest, name, data_buffer, data_byte_size, data_memory_type,
        data_memory_id));
  }

  return nullptr;  // success
}

TRITONSERVER_Error* BLSExecutor::PrepareInferenceOutput(
    const std::vector<const char*>& output_names,
    TRITONSERVER_InferenceRequest* irequest) {
  // Indicate the output tensors to be calculated and returned
  // for the inference request.

  for (const auto& output_name : output_names) {
    RETURN_IF_ERROR(
        TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output_name));
  }

  return nullptr;  // success
}

torch::Tensor BLSExecutor::Execute(std::vector<torch::Tensor>& input_tensors,
                                   std::vector<const char*>& input_names,
                                   std::vector<const char*>& output_names,
                                   std::string model_name) {
  // Check if both models are valid before executing request.
  // Check if the model is ready.
  bool is_ready = false;
  THROW_IF_TRITON_ERROR(TRITONSERVER_ServerModelIsReady(
      server_, model_name.c_str(), -1 /* model_version */, &is_ready));
  if (!is_ready) {
    throw BLSBackendException(
        (std::string("Failed to execute the inference request. Model '") +
         model_name.c_str() + "' is not ready.")
            .c_str());
  }

  // Prepare std::future for model.
  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(1);

  // The inference request object for sending internal requests.
  TRITONSERVER_InferenceRequest* irequest = nullptr;

  try {
    THROW_IF_TRITON_ERROR(PrepareInferenceRequest(&irequest, model_name));
    THROW_IF_TRITON_ERROR(
        PrepareInferenceInput(input_tensors, input_names, irequest));
    THROW_IF_TRITON_ERROR(PrepareInferenceOutput(output_names, irequest));

    // Execute inference request.
    THROW_IF_TRITON_ERROR(model_executor_.AsyncExecute(irequest, &futures[0]));
  } catch (const BLSBackendException& bls_exception) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, bls_exception.what());
    LOG_IF_ERROR(TRITONSERVER_InferenceRequestDelete(irequest),
                 "Failed to delete inference request.");
  }

  // If both internal requests are sent successfully, retrieve the output from
  // each request and construct the final response.
  torch::Tensor r = ConstructFinalResponse(std::move(futures));
  return r;
}

torch::Tensor BLSExecutor::ConstructFinalResponse(
    std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures) {
  std::vector<TRITONSERVER_InferenceResponse*> completed_responses = {nullptr};

  const char* output_name;
  TRITONSERVER_DataType output_datatype;
  const int64_t* output_shape;
  uint64_t dims_count;
  size_t output_byte_size;
  TRITONSERVER_MemoryType output_memory_type;
  int64_t output_memory_id;
  const void* output_base;
  void* userp;
  size_t icount = 0;
  // Retrieve the corresponding TRITONSERVER_InferenceResponse object from
  // 'futures'. The InferResponseComplete function sets the std::promise
  // so that this thread will block until the response is returned.
  completed_responses[icount] = futures[icount].get();
  try {
    THROW_IF_TRITON_ERROR(
        TRITONSERVER_InferenceResponseError(completed_responses[icount]));
  } catch (const BLSBackendException& bls_exception) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, bls_exception.what());

    if (completed_responses[icount] != nullptr) {
      LOG_IF_ERROR(
          TRITONSERVER_InferenceResponseDelete(completed_responses[icount]),
          "Failed to delete inference response.");
    }
  }
  // Retrieve outputs from 'completed_responses'.

  TRITONSERVER_InferenceResponseOutput(
      completed_responses[icount], icount, &output_name, &output_datatype,
      &output_shape, &dims_count, &output_base, &output_byte_size,
      &output_memory_type, &output_memory_id, &userp);

  // TODO: FIX ME, currently put all tensors on cpu.
  auto updated_options =
      torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);

  std::vector<int64_t> batchn_shape(output_shape, output_shape + dims_count);
  torch::Tensor output_tensor = torch::from_blob(const_cast<void*>(output_base),
                                                 batchn_shape, updated_options)
                                    .clone();

  LOG_IF_ERROR(
      TRITONSERVER_InferenceResponseDelete(completed_responses[icount]),
      "Failed to delete inference response.");
  return output_tensor;
}

}  // namespace scorer
}  // namespace backend
}  // namespace triton
