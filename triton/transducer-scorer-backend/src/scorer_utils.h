
#pragma once

#include "torch/script.h"
#include <triton/core/tritonserver.h>

using triton::common::TritonJson;

static void BuildDecoderInput(
    const std::vector<std::vector<int32_t>> &r,
    torch::Tensor *decoder_input) {
  int32_t batch_size = decoder_input->size(0);
  int32_t context_size = decoder_input->size(1);
  int64_t *p = decoder_input->data_ptr<int64_t>();
  for (int32_t i = 0; i != batch_size; ++i) {
    auto start = r[i].end() - context_size;
    auto end = r[i].end();
    std::copy(start, end, p);
    p += context_size;
  }
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

