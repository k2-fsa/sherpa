// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "torch/all.h"
#include "torch/script.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include "bls.h"
#include "scorer_utils.h"
#include "symbol-table.h"




namespace triton { namespace backend { namespace scorer {

static std::string Convert(
    const std::vector<int32_t> &src, const sherpa::SymbolTable * sym_table) {

  std::string text;
  for (auto i : src) {
    auto sym = (*sym_table)[i];
    text.append(sym);
  }
  return text;
}


struct ModelParams {
  std::string decoding_method;
  std::string tokenizer_file;
  
  int context_size;
  uint64_t max_batch_size;

};

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Name of the input and output tensor
  //const std::string& InputTensorName() const { return input_name_; }
  //const std::string& OutputTensorName() const { return output_name_; }

  // Datatype of the input and output tensor
  //TRITONSERVER_DataType TensorDataType() const { return datatype_; }

  // Shape of the input and output tensor as given in the model
  // configuration file. This shape will not include the batch
  // dimension (if the model has one).
  //const std::vector<int64_t>& TensorNonBatchShape() const { return nb_shape_; }

  // Shape of the input and output tensor, including the batch
  // dimension (if the model has one). This method cannot be called
  // until the model is completely loaded and initialized, including
  // all instances of the model. In practice, this means that backend
  // should only call it in TRITONBACKEND_ModelInstanceExecute.
  //TRITONSERVER_Error* TensorShape(std::vector<int64_t>& shape);

  // Validate that this model is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Obtain the parameters parsed from the model configuration
  const ModelParams* Parameters() { return &model_params_; }
  const sherpa::SymbolTable* getSymbolTable() {return &symbol_table_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  ModelParams model_params_;
  sherpa::SymbolTable symbol_table_;

  //std::string input_name_;
  //std::string output_name_;

  //TRITONSERVER_DataType datatype_;

  //bool shape_initialized_;
  //std::vector<int64_t> nb_shape_;
  //std::vector<int64_t> shape_;
};

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
  // Validate that the model's configuration matches what is supported
  // by this backend.
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
  symbol_table_ = sherpa::SymbolTable(model_params_.tokenizer_file);

}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

// TRITONSERVER_Error*
// ModelState::TensorShape(std::vector<int64_t>& shape)
// {
//   // This backend supports models that batch along the first dimension
//   // and those that don't batch. For non-batch models the output shape
//   // will be the shape from the model configuration. For batch models
//   // the output shape will be the shape from the model configuration
//   // prepended with [ -1 ] to represent the batch dimension. The
//   // backend "responder" utility used below will set the appropriate
//   // batch dimension value for each response. The shape needs to be
//   // initialized lazily because the SupportsFirstDimBatching function
//   // cannot be used until the model is completely loaded.
//   if (!shape_initialized_) {
//     bool supports_first_dim_batching;
//     RETURN_IF_ERROR(SupportsFirstDimBatching(&supports_first_dim_batching));
//     if (supports_first_dim_batching) {
//       shape_.push_back(-1);
//     }

//     shape_.insert(shape_.end(), nb_shape_.begin(), nb_shape_.end());
//     shape_initialized_ = true;
//   }

//   shape = shape_;

//   return nullptr;  // success
// }

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // If verbose logging is enabled, dump the model's configuration as
  // JSON into the console output.
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
  }

  // ModelConfig is the model configuration as a TritonJson
  // object. Use the TritonJson utilities to parse the JSON and
  // determine if the configuration is supported by this backend.
  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

  // The model must have exactly 1 input and 1 output.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must have 2 input"));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must have 1 output"));

  RETURN_IF_ERROR(ModelConfig().MemberAsUInt(
                              "max_batch_size", &model_params_.max_batch_size));

 // Validate and set parameters
  common::TritonJson::Value params;
  RETURN_ERROR_IF_FALSE(
      (model_config_.Find("parameters", &params)),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("missing parameters in the model configuration"));
  RETURN_IF_ERROR(ReadParameter(params, "context_size",
                                             &(model_params_.context_size)));
  RETURN_IF_ERROR(ReadParameter(params, "tokenizer_file",
                                             &(model_params_.tokenizer_file)));
  RETURN_IF_ERROR(ReadParameter(params, "decoding_method",
                                             &(model_params_.decoding_method)));
  // common::TritonJson::Value input, output;
  // RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  // RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

  // // Record the input and output name in the model state.
  // const char* input_name;
  // size_t input_name_len;
  // RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
  // input_name_ = std::string(input_name);

  // const char* output_name;
  // size_t output_name_len;
  // RETURN_IF_ERROR(
  //     output.MemberAsString("name", &output_name, &output_name_len));
  // output_name_ = std::string(output_name);

  // // Input and output must have same datatype
  // std::string input_dtype, output_dtype;
  // RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  // RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
  // RETURN_ERROR_IF_FALSE(
  //     input_dtype == output_dtype, TRITONSERVER_ERROR_INVALID_ARG,
  //     std::string("expected input and output datatype to match, got ") +
  //         input_dtype + " and " + output_dtype);
  // datatype_ = ModelConfigDataTypeToTritonServerDataType(input_dtype);

  // // Input and output must have same shape. Reshape is not supported
  // // on either input or output so flag an error is the model
  // // configuration uses it.
  // triton::common::TritonJson::Value reshape;
  // RETURN_ERROR_IF_TRUE(
  //     input.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
  //     std::string("reshape not supported for input tensor"));
  // RETURN_ERROR_IF_TRUE(
  //     output.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
  //     std::string("reshape not supported for output tensor"));

  // std::vector<int64_t> input_shape, output_shape;
  // RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
  // RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

  // RETURN_ERROR_IF_FALSE(
  //     input_shape == output_shape, TRITONSERVER_ERROR_INVALID_ARG,
  //     std::string("expected input and output shape to match, got ") +
  //         backend::ShapeToString(input_shape) + " and " +
  //         backend::ShapeToString(output_shape));

  // nb_shape_ = input_shape;

  return nullptr;  // success
}


/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state),
        bls_executor_(model_state->TritonServer()),
        device_(torch::kCPU)
  {
  #ifdef TRITON_ENABLE_GPU
     device_ = torch::Device(torch::kCUDA, DeviceId());
  #endif
  }

  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      std::vector<torch::jit::IValue>* input_tensors,
      std::vector<BackendMemory*>* input_memories, bool* cuda_copy);

  void SetOutputBuffer(const std::string& out_bytes,
                       TRITONBACKEND_Response* response,
                       TRITONBACKEND_Output* response_output);
  
  std::vector<std::vector<int32_t>> Search(
      std::vector<torch::jit::IValue>* input_tensors
      );


  ModelState* model_state_;
  BLSExecutor bls_executor_;
  torch::Device device_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  //uint64_t exec_start_ns = 0;
  //SET_TIMESTAMP(exec_start_ns);

  const ModelParams* model_params = model_state_->Parameters();

  const int max_batch_size = model_params->max_batch_size;

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to PyTorch backend for '" + Name() + "'")
                  .c_str()));
      return;
    }
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (size_t i = 0; i < request_count; i++) {
    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size.
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed, err);
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if (!all_response_failed) {
    if ((total_batch_size != 1) &&
        (total_batch_size > (size_t)max_batch_size)) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "batch size " + std::to_string(total_batch_size) + " for '" +
                  Name() + "', max allowed is " +
                  std::to_string(max_batch_size))
                  .c_str()));
    }
  }

  std::vector<const char*> input_names;

  std::vector<torch::jit::IValue> input_tensors;
  
  
  std::vector<BackendMemory*> input_memories;
  bool cuda_copy = false;
  std::unique_ptr<BackendInputCollector> collector;
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
// #ifdef TRITON_ENABLE_GPU
//     RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
//         responses, request_count, all_response_failed,
//         ConvertCUDAStatusToTritonError(
//             cudaEventRecord(compute_input_start_event_, stream_),
//             TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
// #endif
  }

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
        CudaStream(), nullptr, nullptr, 0, HostPolicyName().c_str()));
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        SetInputTensors(
            total_batch_size, requests, request_count, &responses,
            collector.get(), &input_names, &input_tensors, &input_memories,
            &cuda_copy));
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
    cuda_copy = false;
  }
#endif

  //std::vector<torch::jit::IValue> output_tensors;
  // uint64_t compute_start_ns = 0;

  // RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
  //     responses, request_count, all_response_failed,
  //     RecordBackendTimestamp(
  //         &compute_start_ns,
  //         reinterpret_cast<void*>(&compute_infer_start_event_)));

  // Run...
  std::vector<std::vector<int32_t>> ans;

  if (!all_response_failed) {
    ans = Search(&input_tensors);
  }

  std::vector<std::string> ans_str;
  const sherpa::SymbolTable * symbol_table = model_state_->getSymbolTable();
  for(auto &utt:ans){
    ans_str.push_back(Convert(utt, symbol_table));
  }





  // for (size_t i = 0; i < request_count; i++) {
  //   // If we get a nullptr request then something is badly wrong. Fail
  //   // and release all requests.
  //   if (requests[i] == nullptr) {
  //     RequestsRespondWithError(
  //         requests, request_count,
  //         TRITONSERVER_ErrorNew(
  //             TRITONSERVER_ERROR_INTERNAL,
  //             std::string(
  //                 "null request given to BLS backend for '" + Name() + "'")
  //                 .c_str()));
  //     return;
  //   }
  // }

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  // std::vector<TRITONBACKEND_Response*> responses;
  // responses.reserve(request_count);

  // for (size_t i = 0; i < request_count; i++) {
  //   TRITONBACKEND_Response* response;
  //   auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
  //   if (err == nullptr) {
  //     responses.emplace_back(response);
  //   } else {
  //     responses.emplace_back(nullptr);
  //     LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
  //     TRITONSERVER_ErrorDelete(err);
  //   }
  // }

  //ModelState* model_state = reinterpret_cast<ModelState*>(Model());

  // The way we collect these batch timestamps is not entirely
  // accurate. Normally, in a performant backend you would execute all
  // the requests at the same time, and so there would be a single
  // compute-start / compute-end time-range. But here we execute each
  // request separately so there is no single range. As a result we
  // just show the entire execute time as being the compute time as
  // well.
  // uint64_t compute_start_ns = 0;
  // SET_TIMESTAMP(compute_start_ns);

  // // Create a BLSExecutor object. To separate from standard backend
  // // implementation, the BLS logic is placed inside class BLSExecutor.
  // BLSExecutor bls_executor(model_state->TritonServer());

  // for (size_t r = 0; r < request_count; r++) {
  //   bls_executor.Execute(requests[r], &responses[r]);
  // }

  // uint64_t compute_end_ns = 0;
  // SET_TIMESTAMP(compute_end_ns);

  // uint64_t exec_end_ns = 0;
  // SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  std::vector<int64_t> output_shape {1,1};
  int dims_count = 2;
  int i = 0;
  for (auto& response : responses) {
    if (response != nullptr) {
      TRITONBACKEND_Output* response_output;
      RESPOND_AND_SET_NULL_IF_ERROR(
        &response, TRITONBACKEND_ResponseOutput(
                      response, &response_output, "OUTPUT0", TRITONSERVER_TYPE_BYTES,
                      &output_shape[0], dims_count));
      SetOutputBuffer(ans_str[i], response, response_output);

      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send BLS backend response");
    }
    i++;
  }

  // // Report statistics for each request.
  // for (uint32_t r = 0; r < request_count; ++r) {
  //   auto& request = requests[r];
  //   LOG_IF_ERROR(
  //       TRITONBACKEND_ModelInstanceReportStatistics(
  //           TritonModelInstance(), request,
  //           (responses[r] != nullptr) /* success */, exec_start_ns,
  //           compute_start_ns, compute_end_ns, exec_end_ns),
  //       "failed reporting request statistics");

  //   LOG_IF_ERROR(
  //       TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
  //       "failed releasing request");
  // }

  // // Report the entire batch statistics.
  // LOG_IF_ERROR(
  //     TRITONBACKEND_ModelInstanceReportBatchStatistics(
  //         TritonModelInstance(), 1 /*total_batch_size*/, exec_start_ns,
  //         compute_start_ns, compute_end_ns, exec_end_ns),
  //     "failed reporting batch request statistics");

  // LOG_MESSAGE(
  //     TRITONSERVER_LOG_VERBOSE,
  //     (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
  //      " released " + std::to_string(request_count) + " requests")
  //         .c_str());
}
//
// Backend that demonstrates the TRITONBACKEND API. This backend works
// for any model that has 1 input with any datatype and any shape and
// 1 output with the same shape and datatype as the input. The backend
// supports both batching and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

void ModelInstanceState::SetOutputBuffer(
    const std::string& out_bytes, TRITONBACKEND_Response* response,
    TRITONBACKEND_Output* response_output) {
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;
  uint32_t byte_size_with_size_int = out_bytes.size() + sizeof(int32_t);
  void* obuffer;  // output buffer
  auto err = TRITONBACKEND_OutputBuffer(
      response_output, &obuffer, byte_size_with_size_int, &actual_memory_type,
      &actual_memory_type_id);
  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(&response, err);
  }

  int32_t* buffer_as_int = reinterpret_cast<int32_t*>(obuffer);
  buffer_as_int[0] = out_bytes.size();
  memcpy(&buffer_as_int[1], out_bytes.data(), out_bytes.size());
}
/////////////
std::vector<std::vector<int32_t>>
ModelInstanceState::Search(
    std::vector<torch::jit::IValue>* input_tensors
    )
{
  //torch::NoGradGuard no_grad;
  torch::Tensor encoder_out, encoder_out_length;
  encoder_out = (*input_tensors)[0].toTensor();
  encoder_out_length = (*input_tensors)[1].toTensor();
  encoder_out_length = encoder_out_length.to(torch::kCPU);

  TORCH_CHECK(encoder_out.dim() == 3, "encoder_out.dim() is ",
              encoder_out.dim(), "Expected value is 3");
  //TORCH_CHECK(encoder_out.scalar_type() == torch::kFloat,
   //           "encoder_out.scalar_type() is ", encoder_out.scalar_type());

  TORCH_CHECK(encoder_out_length.dim() == 1, "encoder_out_length.dim() is",
              encoder_out_length.dim());
  //TORCH_CHECK(encoder_out_length.scalar_type() == torch::kLong,
  //           "encoder_out_length.scalar_type() is ",
  //          encoder_out_length.scalar_type());

  TORCH_CHECK(encoder_out_length.device().is_cpu());

  //torch::Device device = model_->Device();

  torch::nn::utils::rnn::PackedSequence packed_seq =
      torch::nn::utils::rnn::pack_padded_sequence(encoder_out,
                                                  encoder_out_length,
                                                  /*batch_first*/ true,
                                                  /*enforce_sorted*/ false);

  int32_t blank_id = 0;  // hard-code
  int32_t context_size = 2; // hard-code for now , TOOD: yuekai 

  int32_t N = encoder_out_length.size(0);

  std::vector<int32_t> padding(context_size, blank_id);
  std::vector<std::vector<int32_t>> results(N, padding);

  // for (auto &r : results) {
  //   // We will remove the padding at the end
  //   r.tokens = padding;
  // }

  auto decoder_input =
      torch::full({N, context_size}, blank_id,
                  torch::dtype(torch::kLong)
                      .memory_format(torch::MemoryFormat::Contiguous));

  // its shape is (N, 1, joiner_dim)
  std::string decoder_name = "decoder";
  std::string joiner_name = "joiner";
  std::vector<const char*> decoder_input_name {"y"};
  std::vector<const char*> decoder_output_name {"decoder_out"};
  std::vector<const char*> joiner_input_name {"encoder_out", "decoder_out"};
  std::vector<const char*> joiner_output_name {"logit"};
  std::vector<torch::Tensor> decoder_input_tensors {decoder_input.to(device_)};
  auto decoder_out = bls_executor_.Execute(decoder_input_tensors, decoder_input_name, decoder_output_name, decoder_name);
  //auto decoder_out = model_->RunDecoder(decoder_input.to(device));

  using torch::indexing::Slice;
  auto batch_sizes_accessor = packed_seq.batch_sizes().accessor<int64_t, 1>();

  int32_t max_T = packed_seq.batch_sizes().numel();

  int32_t offset = 0;
  for (int32_t t = 0; t != max_T; ++t) {
    int32_t cur_batch_size = batch_sizes_accessor[t];
    int32_t start = offset;
    int32_t end = start + cur_batch_size;
    auto cur_encoder_out = packed_seq.data().index({Slice(start, end)});
    offset = end;

    // cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // Now cur_encoder_out is of shape (cur_batch_size, 1, 1, joiner_dim)
    if (cur_batch_size < decoder_out.size(0)) {
      decoder_out = decoder_out.index({Slice(0, cur_batch_size)});
    }

    // auto logits = model_->RunJoiner(cur_encoder_out, decoder_out.unsqueeze(1));
    
    // auto logits = model_->RunJoiner(cur_encoder_out, decoder_out.squeeze(1));
     std::vector<torch::Tensor> joiner_input_tensors {cur_encoder_out, decoder_out.squeeze(1)};
     auto logits = bls_executor_.Execute(joiner_input_tensors, joiner_input_name, joiner_output_name, joiner_name);
    
    
    // logits' shape is (cur_batch_size, 1, 1, vocab_size)
    // logits is the output of nn.Linear. Since we are using greedy search
    // and only the magnitude matters, we don't invoke log_softmax here

    // logits = logits.squeeze(1).squeeze(1);
    auto max_indices = logits.argmax(/*dim*/ -1).cpu();
    auto max_indices_accessor = max_indices.accessor<int64_t, 1>();
    bool emitted = false;
    for (int32_t k = 0; k != cur_batch_size; ++k) {
      auto index = max_indices_accessor[k];
      if (index != blank_id) {
        emitted = true;
        results[k].push_back(index);
        //results[k].tokens.push_back(index);
        //results[k].timestamps.push_back(t);
      }
    }

    if (emitted) {
      // BuildDecoderInput(results, &(decoder_input.to(torch::kCPU)));
      BuildDecoderInput(results, &decoder_input);
      std::vector<torch::Tensor> decoder_input_tensors {decoder_input.to(device_)};
      auto decoder_out = bls_executor_.Execute(decoder_input_tensors, decoder_input_name, decoder_output_name, decoder_name);
    }
  }  // for (int32_t t = 0; t != max_T; ++t) {

  auto unsorted_indices = packed_seq.unsorted_indices().cpu();
  auto unsorted_indices_accessor = unsorted_indices.accessor<int64_t, 1>();

  // std::vector<OfflineTransducerDecoderResult> ans(N);
  std::vector<std::vector<int32_t>> ans(N);

  for (int32_t i = 0; i != N; ++i) {
    int32_t k = unsorted_indices_accessor[i];
    torch::ArrayRef<int32_t> arr(results[k]);
    // torch::ArrayRef<int32_t> arr(results[k].tokens);
    ans[i] = arr.slice(context_size).vec();
    // ans[i].tokens = arr.slice(context_size).vec();
    //ans[i].timestamps = std::move(results[k].timestamps);
  }

  return ans;
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<torch::jit::IValue>* input_tensors,
    std::vector<BackendMemory*>* input_memories, bool* cuda_copy)
{
  // InferenceMode should be used to guard all tensors operations
  // torch::InferenceMode infer_guard(model_state_->EnabledInferenceMode());

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  input_tensors->resize(input_count);
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    input_names->emplace_back(input_name);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);
  
    batchn_shape[0] = total_batch_size;

    // The input must be in contiguous CPU/GPU memory.
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    if (device_.is_cpu()) {
      alloc_perference = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                          {TRITONSERVER_MEMORY_CPU, 0}};
    } else {
      alloc_perference = {{TRITONSERVER_MEMORY_GPU, device_.index()}};
    }

    const char* input_buffer;
    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    RETURN_IF_ERROR(collector->ProcessTensor(
        input_name, nullptr, 0, alloc_perference, &input_buffer,
        &batchn_byte_size, &memory_type, &memory_type_id));

    // Create Torch tensor
    // const auto torch_dtype = ConvertDataTypeToTorchType(input_datatype);
    // torch::TensorOptions options{torch_dtype.second};
    // auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
    //                            ? options.device(torch::kCUDA, device_.index())
    //                            : options.device(torch::kCPU);

    auto updated_options =
    torch::TensorOptions()
      .dtype(torch::kHalf)
      .device(torch::kCUDA, 0);

    torch::Tensor input_tensor = torch::from_blob(
    const_cast<char*>(input_buffer), batchn_shape, updated_options);
    input_tensors->push_back(input_tensor);
    //(*input_tensors)[input_index_map_[input_name]] = input_tensor;

    // if (input_datatype == TRITONSERVER_TYPE_BYTES) {
    //   // Create the PyTorch list to hold the strings.
    //   torch::List<std::string> input_list;
    //   input_list.reserve(batchn_shape[0]);

    //   for (size_t idx = 0; idx < request_count; idx++) {
    //     TRITONBACKEND_Input* input;
    //     RESPOND_AND_SET_NULL_IF_ERROR(
    //         &((*responses)[idx]),
    //         TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
    //     const int64_t* shape;
    //     uint32_t dims_count;
    //     uint32_t buffer_count;
    //     RESPOND_AND_SET_NULL_IF_ERROR(
    //         &((*responses)[idx]),
    //         TRITONBACKEND_InputPropertiesForHostPolicy(
    //             input, HostPolicyName().c_str(), nullptr, nullptr, &shape,
    //             &dims_count, nullptr, &buffer_count));

    //     const int64_t batch_element_cnt = GetElementCount(shape, dims_count);

    //     *cuda_copy |= SetStringInputTensor(
    //         &input_list, input, input_name, buffer_count, batch_element_cnt,
    //         &((*responses)[idx]), CudaStream(), HostPolicyName().c_str());
    //   }

    //   (*input_tensors)[input_index_map_[input_name]] = input_list;
    // } else {
    //   // Remove constness to align with the signature of torch::from_blob()
    //   torch::Tensor input_tensor = torch::from_blob(
    //       const_cast<char*>(input_buffer), batchn_shape, updated_options);
    //   (*input_tensors)[input_index_map_[input_name]] = input_tensor;
    // }
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();

  return nullptr;
}

extern "C" {

// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // This backend does not require any "global" state but as an
  // example create a string to demonstrate.
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  // Delete the "global" state associated with the backend.
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

}  // extern "C"



extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  // extern "C"



extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Collect various timestamps during the execution of this batch or
  // requests. These values are reported below before returning from
  // the function.

  //uint64_t exec_start_ns = 0;
  //SET_TIMESTAMP(exec_start_ns);

  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Best practice for a high-performance
  // implementation is to avoid introducing mutex/lock and instead use
  // only function-local and model-instance-specific state.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  
  // ModelState* model_state = instance_state->StateForModel();

  instance_state->ProcessRequests(requests, request_count);

  // 'responses' is initialized as a parallel array to 'requests',
  // with one TRITONBACKEND_Response object for each
  // TRITONBACKEND_Request object. If something goes wrong while
  // creating these response objects, the backend simply returns an
  // error from TRITONBACKEND_ModelInstanceExecute, indicating to
  // Triton that this backend did not create or send any responses and
  // so it is up to Triton to create and send an appropriate error
  // response for each request. RETURN_IF_ERROR is one of several
  // useful macros for error handling that can be found in
  // backend_common.h.

//   std::vector<TRITONBACKEND_Response*> responses;
//   responses.reserve(request_count);
//   for (uint32_t r = 0; r < request_count; ++r) {
//     TRITONBACKEND_Request* request = requests[r];
//     TRITONBACKEND_Response* response;
//     RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
//     responses.push_back(response);
//   }

//   // At this point, the backend takes ownership of 'requests', which
//   // means that it is responsible for sending a response for every
//   // request. From here, even if something goes wrong in processing,
//   // the backend must return 'nullptr' from this function to indicate
//   // success. Any errors and failures must be communicated via the
//   // response objects.
//   //
//   // To simplify error handling, the backend utilities manage
//   // 'responses' in a specific way and it is recommended that backends
//   // follow this same pattern. When an error is detected in the
//   // processing of a request, an appropriate error response is sent
//   // and the corresponding TRITONBACKEND_Response object within
//   // 'responses' is set to nullptr to indicate that the
//   // request/response has already been handled and no futher processing
//   // should be performed for that request. Even if all responses fail,
//   // the backend still allows execution to flow to the end of the
//   // function so that statistics are correctly reported by the calls
//   // to TRITONBACKEND_ModelInstanceReportStatistics and
//   // TRITONBACKEND_ModelInstanceReportBatchStatistics.
//   // RESPOND_AND_SET_NULL_IF_ERROR, and
//   // RESPOND_ALL_AND_SET_NULL_IF_ERROR are macros from
//   // backend_common.h that assist in this management of response
//   // objects.

//   // The backend could iterate over the 'requests' and process each
//   // one separately. But for performance reasons it is usually
//   // preferred to create batched input tensors that are processed
//   // simultaneously. This is especially true for devices like GPUs
//   // that are capable of exploiting the large amount parallelism
//   // exposed by larger data sets.
//   //
//   // The backend utilities provide a "collector" to facilitate this
//   // batching process. The 'collector's ProcessTensor function will
//   // combine a tensor's value from each request in the batch into a
//   // single contiguous buffer. The buffer can be provided by the
//   // backend or 'collector' can create and manage it. In this backend,
//   // there is not a specific buffer into which the batch should be
//   // created, so use ProcessTensor arguments that cause collector to
//   // manage it. ProcessTensor does NOT support TRITONSERVER_TYPE_BYTES
//   // data type.

//   BackendInputCollector collector(
//       requests, request_count, &responses, model_state->TritonMemoryManager(),
//       false /* pinned_enabled */, nullptr /* stream*/);

//   // To instruct ProcessTensor to "gather" the entire batch of input
//   // tensors into a single contiguous buffer in CPU memory, set the
//   // "allowed input types" to be the CPU ones (see tritonserver.h in
//   // the triton-inference-server/core repo for allowed memory types).
//   std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types =
//       {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

//   const char* input_buffer;
//   size_t input_buffer_byte_size;
//   TRITONSERVER_MemoryType input_buffer_memory_type;
//   int64_t input_buffer_memory_type_id;

//   RESPOND_ALL_AND_SET_NULL_IF_ERROR(
//       responses, request_count,
//       collector.ProcessTensor(
//           model_state->InputTensorName().c_str(), nullptr /* existing_buffer */,
//           0 /* existing_buffer_byte_size */, allowed_input_types, &input_buffer,
//           &input_buffer_byte_size, &input_buffer_memory_type,
//           &input_buffer_memory_type_id));

//   // Finalize the collector. If 'true' is returned, 'input_buffer'
//   // will not be valid until the backend synchronizes the CUDA
//   // stream or event that was used when creating the collector. For
//   // this backend, GPU is not supported and so no CUDA sync should
//   // be needed; so if 'true' is returned simply log an error.
//   const bool need_cuda_input_sync = collector.Finalize();
//   if (need_cuda_input_sync) {
//     LOG_MESSAGE(
//         TRITONSERVER_LOG_ERROR,
//         "'recommended' backend: unexpected CUDA sync required by collector");
//   }

//   // 'input_buffer' contains the batched input tensor. The backend can
//   // implement whatever logic is necessary to produce the output
//   // tensor. This backend simply logs the input tensor value and then
//   // returns the input tensor value in the output tensor so no actual
//   // computation is needed.

//   uint64_t compute_start_ns = 0;
//   SET_TIMESTAMP(compute_start_ns);

//   LOG_MESSAGE(
//       TRITONSERVER_LOG_INFO,
//       (std::string("model ") + model_state->Name() + ": requests in batch " +
//        std::to_string(request_count))
//           .c_str());
//   std::string tstr;
//   IGNORE_ERROR(BufferAsTypedString(
//       tstr, input_buffer, input_buffer_byte_size,
//       model_state->TensorDataType()));
//   LOG_MESSAGE(
//       TRITONSERVER_LOG_INFO,
//       (std::string("batched " + model_state->InputTensorName() + " value: ") +
//        tstr)
//           .c_str());

//   const char* output_buffer = input_buffer;
//   TRITONSERVER_MemoryType output_buffer_memory_type = input_buffer_memory_type;
//   int64_t output_buffer_memory_type_id = input_buffer_memory_type_id;

//   uint64_t compute_end_ns = 0;
//   SET_TIMESTAMP(compute_end_ns);

//   bool supports_first_dim_batching;
//   RESPOND_ALL_AND_SET_NULL_IF_ERROR(
//       responses, request_count,
//       model_state->SupportsFirstDimBatching(&supports_first_dim_batching));

//   std::vector<int64_t> tensor_shape;
//   RESPOND_ALL_AND_SET_NULL_IF_ERROR(
//       responses, request_count, model_state->TensorShape(tensor_shape));

//   // Because the output tensor values are concatenated into a single
//   // contiguous 'output_buffer', the backend must "scatter" them out
//   // to the individual response output tensors.  The backend utilities
//   // provide a "responder" to facilitate this scattering process.
//   // BackendOutputResponder does NOT support TRITONSERVER_TYPE_BYTES
//   // data type.

//   // The 'responders's ProcessTensor function will copy the portion of
//   // 'output_buffer' corresonding to each request's output into the
//   // response for that request.

//   BackendOutputResponder responder(
//       requests, request_count, &responses, model_state->TritonMemoryManager(),
//       supports_first_dim_batching, false /* pinned_enabled */,
//       nullptr /* stream*/);

//   responder.ProcessTensor(
//       model_state->OutputTensorName().c_str(), model_state->TensorDataType(),
//       tensor_shape, output_buffer, output_buffer_memory_type,
//       output_buffer_memory_type_id);

//   // Finalize the responder. If 'true' is returned, the output
//   // tensors' data will not be valid until the backend synchronizes
//   // the CUDA stream or event that was used when creating the
//   // responder. For this backend, GPU is not supported and so no CUDA
//   // sync should be needed; so if 'true' is returned simply log an
//   // error.
//   const bool need_cuda_output_sync = responder.Finalize();
//   if (need_cuda_output_sync) {
//     LOG_MESSAGE(
//         TRITONSERVER_LOG_ERROR,
//         "'recommended' backend: unexpected CUDA sync required by responder");
//   }

//   // Send all the responses that haven't already been sent because of
//   // an earlier error.
//   for (auto& response : responses) {
//     if (response != nullptr) {
//       LOG_IF_ERROR(
//           TRITONBACKEND_ResponseSend(
//               response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
//           "failed to send response");
//     }
//   }

//   uint64_t exec_end_ns = 0;
//   SET_TIMESTAMP(exec_end_ns);

// #ifdef TRITON_ENABLE_STATS
//   // For batch statistics need to know the total batch size of the
//   // requests. This is not necessarily just the number of requests,
//   // because if the model supports batching then any request can be a
//   // batched request itself.
//   size_t total_batch_size = 0;
//   if (!supports_first_dim_batching) {
//     total_batch_size = request_count;
//   } else {
//     for (uint32_t r = 0; r < request_count; ++r) {
//       auto& request = requests[r];
//       TRITONBACKEND_Input* input = nullptr;
//       LOG_IF_ERROR(
//           TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
//           "failed getting request input");
//       if (input != nullptr) {
//         const int64_t* shape = nullptr;
//         LOG_IF_ERROR(
//             TRITONBACKEND_InputProperties(
//                 input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr),
//             "failed getting input properties");
//         if (shape != nullptr) {
//           total_batch_size += shape[0];
//         }
//       }
//     }
//   }
// #else
//   (void)exec_start_ns;
//   (void)exec_end_ns;
//   (void)compute_start_ns;
//   (void)compute_end_ns;
// #endif  // TRITON_ENABLE_STATS

//   // Report statistics for each request, and then release the request.
//   for (uint32_t r = 0; r < request_count; ++r) {
//     auto& request = requests[r];

// #ifdef TRITON_ENABLE_STATS
//     LOG_IF_ERROR(
//         TRITONBACKEND_ModelInstanceReportStatistics(
//             instance_state->TritonModelInstance(), request,
//             (responses[r] != nullptr) /* success */, exec_start_ns,
//             compute_start_ns, compute_end_ns, exec_end_ns),
//         "failed reporting request statistics");
// #endif  // TRITON_ENABLE_STATS

//     LOG_IF_ERROR(
//         TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
//         "failed releasing request");
//   }

// #ifdef TRITON_ENABLE_STATS
//   // Report batch statistics.
//   LOG_IF_ERROR(
//       TRITONBACKEND_ModelInstanceReportBatchStatistics(
//           instance_state->TritonModelInstance(), total_batch_size,
//           exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
//       "failed reporting batch request statistics");
// #endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::recommended
