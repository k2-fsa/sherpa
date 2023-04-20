
#pragma once

#include <triton/core/tritonserver.h>

#include "symbol-table.h"
#include "torch/script.h"

using triton::common::TritonJson;

namespace triton {
namespace backend {
namespace scorer {

static std::string Convert(const std::vector<int32_t>& src,
                           const sherpa::SymbolTable* sym_table) {
  std::string text;
  for (auto i : src) {
    auto sym = (*sym_table)[i];
    text.append(sym);
  }
  return text;
}

static void BuildDecoderInput(const std::vector<std::vector<int32_t>>& r,
                              torch::Tensor* decoder_input) {
  int32_t batch_size = decoder_input->size(0);
  int32_t context_size = decoder_input->size(1);
  int64_t* p = decoder_input->data_ptr<int64_t>();
  for (int32_t i = 0; i != batch_size; ++i) {
    auto start = r[i].end() - context_size;
    auto end = r[i].end();
    std::copy(start, end, p);
    p += context_size;
  }
}

std::pair<bool, torch::ScalarType> ConvertDataTypeToTorchType(
    const TRITONSERVER_DataType dtype) {
  torch::ScalarType type = torch::kInt;
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      type = torch::kBool;
      break;
    case TRITONSERVER_TYPE_UINT8:
      type = torch::kByte;
      break;
    case TRITONSERVER_TYPE_INT8:
      type = torch::kChar;
      break;
    case TRITONSERVER_TYPE_INT16:
      type = torch::kShort;
      break;
    case TRITONSERVER_TYPE_INT32:
      type = torch::kInt;
      break;
    case TRITONSERVER_TYPE_INT64:
      type = torch::kLong;
      break;
    case TRITONSERVER_TYPE_FP16:
      type = torch::kHalf;
      break;
    case TRITONSERVER_TYPE_FP32:
      type = torch::kFloat;
      break;
    case TRITONSERVER_TYPE_FP64:
      type = torch::kDouble;
      break;
    case TRITONSERVER_TYPE_UINT16:
    case TRITONSERVER_TYPE_UINT32:
    case TRITONSERVER_TYPE_UINT64:
    case TRITONSERVER_TYPE_BYTES:
    default:
      return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, std::string* param) {
  TritonJson::Value value;
  RETURN_ERROR_IF_FALSE(
      params.Find(key.c_str(), &value), TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration is missing the parameter ") + key);
  RETURN_IF_ERROR(value.MemberAsString("string_value", param));
  return nullptr;  // success
}

TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, int* param) {
  std::string tmp;
  RETURN_IF_ERROR(ReadParameter(params, key, &tmp));
  *param = std::stoi(tmp);
  return nullptr;  // success
}

TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, float* param) {
  std::string tmp;
  RETURN_IF_ERROR(ReadParameter(params, key, &tmp));
  *param = std::stof(tmp);
  return nullptr;  // success
}

#ifdef TRITON_ENABLE_GPU
TRITONSERVER_Error* ConvertCUDAStatusToTritonError(cudaError_t cuda_error,
                                                   TRITONSERVER_Error_Code code,
                                                   const char* msg) {
  if (cuda_error != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        code,
        (std::string(msg) + ": " + cudaGetErrorString(cuda_error)).c_str());
  }
  return nullptr;  // success
}
#endif

}  // namespace scorer
}  // namespace backend
}  // namespace triton