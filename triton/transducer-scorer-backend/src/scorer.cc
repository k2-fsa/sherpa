// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "bls.h"
#include "scorer_utils.h"
#include "symbol-table.h"
#include "torch/all.h"
#include "torch/script.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/nvtx.h"
#include "triton/core/tritonbackend.h"

namespace triton {
namespace backend {
namespace scorer {

struct ModelParams {
  std::string decoding_method;
  std::string tokenizer_file;
  int context_size;
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
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model,
                                    ModelState** state);
  virtual ~ModelState() = default;

  // Validate and parse the model configuration
  TRITONSERVER_Error* ValidateModelConfig();

  // Obtain the parameters parsed from the model configuration
  const ModelParams* Parameters() { return &model_params_; }
  const sherpa::SymbolTable* getSymbolTable() { return &symbol_table_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  ModelParams model_params_;
  sherpa::SymbolTable symbol_table_;
};

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model) {
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
  symbol_table_ = sherpa::SymbolTable(model_params_.tokenizer_file);
}

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model,
                                       ModelState** state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error* ModelState::ValidateModelConfig() {
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
  RETURN_ERROR_IF_FALSE(inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("model configuration must have 2 input"));
  RETURN_ERROR_IF_FALSE(outputs.ArraySize() == 1,
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("model configuration must have 1 output"));

  // Validate and set parameters
  common::TritonJson::Value params;
  RETURN_ERROR_IF_FALSE(
      (ModelConfig().Find("parameters", &params)),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("missing parameters in the model configuration"));
  RETURN_IF_ERROR(
      ReadParameter(params, "context_size", &(model_params_.context_size)));
  RETURN_IF_ERROR(
      ReadParameter(params, "tokenizer_file", &(model_params_.tokenizer_file)));
  RETURN_IF_ERROR(ReadParameter(params, "decoding_method",
                                &(model_params_.decoding_method)));
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
  void ProcessRequests(TRITONBACKEND_Request** requests,
                       const uint32_t request_count);

 private:
  ModelInstanceState(ModelState* model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state),
        bls_executor_(model_state->TritonServer()),
        device_(torch::kCPU) {
#ifdef TRITON_ENABLE_GPU
    device_ = torch::Device(torch::kCUDA, DeviceId());
    // Need to set the CUDA context so that the context that events are
    // created on match with contexts that events are recorded with.
    THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
        cudaSetDevice(DeviceId()), TRITONSERVER_ERROR_INTERNAL,
        "Failed to set the device"));
    THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventCreate(&compute_input_start_event_),
        TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
    THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventCreate(&compute_infer_start_event_),
        TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
    THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventCreate(&compute_output_start_event_),
        TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
#endif
    // TODO: FIX this hard code
    input_index_map_["encoder_out"] = 0;
    input_index_map_["encoder_out_lens"] = 1;
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
  TRITONSERVER_Error* RecordBackendTimestamp(uint64_t* timestamp,
                                             void* cuda_event);
  std::vector<std::vector<int32_t>> Search(
      std::vector<torch::jit::IValue>* input_tensors);

  ModelState* model_state_;
  BLSExecutor bls_executor_;
  torch::Device device_;
  std::unordered_map<std::string, int> input_index_map_;

  cudaEvent_t compute_input_start_event_;
  cudaEvent_t compute_infer_start_event_;
  cudaEvent_t compute_output_start_event_;
};

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error* ModelInstanceState::RecordBackendTimestamp(
    uint64_t* timestamp, void* cuda_event) {
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    cudaEvent_t* lcuda_event = reinterpret_cast<cudaEvent_t*>(cuda_event);
    RETURN_IF_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventRecord(*lcuda_event, stream_), TRITONSERVER_ERROR_INTERNAL,
        "Failed to record the event."));
#endif
  } else {
    SET_TIMESTAMP(*timestamp);
  }
  return nullptr;
}

void ModelInstanceState::ProcessRequests(TRITONBACKEND_Request** requests,
                                         const uint32_t request_count) {
  const int max_batch_size = model_state_->MaxBatchSize();

  NVTX_RANGE(nvtx_, "ProcessRequests " + Name());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

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
              std::string("null request given to PyTorch backend for '" +
                          Name() + "'")
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
        err = TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape,
                                            nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count,
                                          all_response_failed, err);
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
              std::string("batch size " + std::to_string(total_batch_size) +
                          " for '" + Name() + "', max allowed is " +
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
#ifdef TRITON_ENABLE_GPU
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(compute_input_start_event_, stream_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  }

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
        CudaStream(), nullptr, nullptr, 0, HostPolicyName().c_str()));

    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        SetInputTensors(total_batch_size, requests, request_count, &responses,
                        collector.get(), &input_names, &input_tensors,
                        &input_memories, &cuda_copy));
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
    cuda_copy = false;
  }
#endif

  uint64_t compute_start_ns = 0;
  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      RecordBackendTimestamp(
          &compute_start_ns,
          reinterpret_cast<void*>(&compute_infer_start_event_)));

  // Run...
  std::vector<std::vector<int32_t>> ans;

  if (!all_response_failed) {
    ans = Search(&input_tensors);
  }

  std::vector<std::string> ans_str;
  const sherpa::SymbolTable* symbol_table = model_state_->getSymbolTable();

  for (auto& utt : ans) {
    ans_str.push_back(Convert(utt, symbol_table));
  }

  uint64_t compute_end_ns = 0;
  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      RecordBackendTimestamp(
          &compute_end_ns,
          reinterpret_cast<void*>(&compute_output_start_event_)));

  std::vector<int64_t> output_shape{1, 1};
  int dims_count = 2;
  int i = 0;
  for (auto& response : responses) {
    if (response != nullptr) {
      TRITONBACKEND_Output* response_output;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response,
          TRITONBACKEND_ResponseOutput(response, &response_output, "OUTPUT0",
                                       TRITONSERVER_TYPE_BYTES,
                                       &output_shape[0], dims_count));
      SetOutputBuffer(ans_str[i], response, response_output);

      LOG_IF_ERROR(TRITONBACKEND_ResponseSend(
                       response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                   "failed to send BLS backend response");
    }
    i++;
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_GPU
  // We have to always synchronize the stream. This is to make sure that
  // the events on the cuda stream are synchronized. Otherwise, the events
  // are only guaranteed to be synchronized if the model provides the output
  // on GPU.
  cudaStreamSynchronize(stream_);
#endif

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    // [FIXME] in the case of cudaEventElapsedTime failure, should handle
    // stats reporting more gracefully as the durations are inaccurate
    float compute_input_duration = 0;
    float compute_infer_duration = 0;
    LOG_IF_ERROR(
        ConvertCUDAStatusToTritonError(
            cudaEventElapsedTime(&compute_input_duration,
                                 compute_input_start_event_,
                                 compute_infer_start_event_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time"),
        "Failed to capture elapsed time");

    LOG_IF_ERROR(
        ConvertCUDAStatusToTritonError(
            cudaEventElapsedTime(&compute_infer_duration,
                                 compute_infer_start_event_,
                                 compute_output_start_event_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time"),
        "Failed to capture elapsed time");

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + (compute_infer_duration * 1e6);
#endif
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(
                     TritonModelInstance(), request,
                     (responses[r] != nullptr) /* success */, exec_start_ns,
                     compute_start_ns, compute_end_ns, exec_end_ns),
                 "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                   TritonModelInstance(), total_batch_size, exec_start_ns,
                   compute_start_ns, compute_end_ns, exec_end_ns),
               "failed reporting batch request statistics");

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
               " released " + std::to_string(request_count) + " requests")
                  .c_str());
}

void ModelInstanceState::SetOutputBuffer(
    const std::string& out_bytes, TRITONBACKEND_Response* response,
    TRITONBACKEND_Output* response_output) {
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;
  uint32_t byte_size_with_size_int = out_bytes.size() + sizeof(int32_t);
  void* obuffer;
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
std::vector<std::vector<int32_t>> ModelInstanceState::Search(
    std::vector<torch::jit::IValue>* input_tensors) {
  NVTX_RANGE(nvtx_, "greedy search " + Name());
  torch::Tensor encoder_out, encoder_out_length;
  encoder_out = (*input_tensors)[0].toTensor();
  encoder_out_length = (*input_tensors)[1].toTensor();
  encoder_out_length = encoder_out_length.to(torch::kCPU);

  TORCH_CHECK(encoder_out.dim() == 3, "encoder_out.dim() is ",
              encoder_out.dim(), "Expected value is 3");
  // Only support fp16 for now
  TORCH_CHECK(encoder_out.scalar_type() == torch::kHalf,
              "encoder_out.scalar_type() is ", encoder_out.scalar_type());

  TORCH_CHECK(encoder_out_length.dim() == 1, "encoder_out_length.dim() is",
              encoder_out_length.dim());
  TORCH_CHECK(encoder_out_length.scalar_type() == torch::kLong,
              "encoder_out_length.scalar_type() is ",
              encoder_out_length.scalar_type());

  TORCH_CHECK(encoder_out_length.device().is_cpu());

  torch::nn::utils::rnn::PackedSequence packed_seq =
      torch::nn::utils::rnn::pack_padded_sequence(encoder_out,
                                                  encoder_out_length,
                                                  /*batch_first*/ true,
                                                  /*enforce_sorted*/ false);

  int32_t blank_id = 0;  // hard-code for now , TOOD: yuekai
  int32_t context_size = model_state_->Parameters()->context_size;

  int32_t N = encoder_out_length.size(0);

  std::vector<int32_t> padding(context_size, blank_id);
  std::vector<std::vector<int32_t>> results(N, padding);

  auto decoder_input =
      torch::full({N, context_size}, blank_id,
                  torch::dtype(torch::kLong)
                      .memory_format(torch::MemoryFormat::Contiguous));

  std::string decoder_name = "decoder";
  std::vector<const char*> decoder_input_name{"y"};
  std::vector<const char*> decoder_output_name{"decoder_out"};
  std::vector<torch::Tensor> decoder_input_tensors{decoder_input.to(device_)};

  auto decoder_out =
      bls_executor_.Execute(decoder_input_tensors, decoder_input_name,
                            decoder_output_name, decoder_name);

  std::string joiner_name = "joiner";
  std::vector<const char*> joiner_input_name{"encoder_out", "decoder_out"};
  std::vector<const char*> joiner_output_name{"logit"};

  using torch::indexing::Slice;
  auto batch_sizes_accessor = packed_seq.batch_sizes().accessor<int64_t, 1>();

  int32_t max_T = packed_seq.batch_sizes().numel();

  int32_t offset = 0;
  for (int32_t t = 0; t != max_T; ++t) {
    int32_t cur_batch_size = batch_sizes_accessor[t];
    int32_t start = offset;
    int32_t end = start + cur_batch_size;
    auto cur_encoder_out = packed_seq.data().index({Slice(start, end)});
    // Now cur_encoder_out is of shape (cur_batch_size, joiner_dim)
    offset = end;

    if (cur_batch_size < decoder_out.size(0)) {
      decoder_out = decoder_out.index({Slice(0, cur_batch_size)});
    }
    std::vector<torch::Tensor> joiner_input_tensors{
        cur_encoder_out, decoder_out.squeeze(1).to(device_)};

    auto logits = bls_executor_.Execute(joiner_input_tensors, joiner_input_name,
                                        joiner_output_name, joiner_name);

    // logits' shape is (cur_batch_size, vocab_size)
    // logits is the output of nn.Linear. Since we are using greedy search
    // and only the magnitude matters, we don't invoke log_softmax here
    auto max_indices = logits.argmax(/*dim*/ -1).cpu();
    auto max_indices_accessor = max_indices.accessor<int64_t, 1>();
    bool emitted = false;
    for (int32_t k = 0; k != cur_batch_size; ++k) {
      auto index = max_indices_accessor[k];
      if (index != blank_id) {
        emitted = true;
        results[k].push_back(index);
        // TODO: add timestamps here
        // results[k].tokens.push_back(index);
        // results[k].timestamps.push_back(t);
      }
    }

    if (emitted) {
      BuildDecoderInput(results, &decoder_input);
      std::vector<torch::Tensor> decoder_input_tensors{
          decoder_input.to(device_)};
      decoder_out =
          bls_executor_.Execute(decoder_input_tensors, decoder_input_name,
                                decoder_output_name, decoder_name);
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
    // ans[i].timestamps = std::move(results[k].timestamps);
  }

  return ans;
}

TRITONSERVER_Error* ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<torch::jit::IValue>* input_tensors,
    std::vector<BackendMemory*>* input_memories, bool* cuda_copy) {
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
    std::vector<int64_t> batchn_shape(input_shape,
                                      input_shape + input_dims_count);

    batchn_shape[0] = total_batch_size;

    // The input must be in contiguous CPU/GPU memory.
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    if (device_.is_cpu()) {
      alloc_perference = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                          {TRITONSERVER_MEMORY_CPU, 0}};
      LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                  (std::string("device is cpu")).c_str());
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
    const auto torch_dtype = ConvertDataTypeToTorchType(input_datatype);
    torch::TensorOptions options{torch_dtype.second};
    auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
                               ? options.device(torch::kCUDA, device_.index())
                               : options.device(torch::kCPU);

    torch::Tensor input_tensor = torch::from_blob(
        const_cast<char*>(input_buffer), batchn_shape, updated_options);
    (*input_tensors)[input_index_map_[input_name]] = input_tensor;
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();

  return nullptr;
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInitialize: ") + name +
               " (version " + std::to_string(version) + ")")
                  .c_str());
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
TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
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
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
               " (" + TRITONSERVER_InstanceGroupKindString(kind) + " device " +
               std::to_string(device_id) + ")")
                  .c_str());

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
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
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
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
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

  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}  // namespace scorer
}  // namespace backend
}  // namespace triton
