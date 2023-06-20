// sherpa/cpp_api/offline-recognizer.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_IMPL_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_IMPL_H_

#include <memory>
#include <vector>

#include "sherpa/cpp_api/offline-recognizer.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

class OfflineRecognizerImpl {
 public:
  virtual ~OfflineRecognizerImpl() = default;

  virtual std::unique_ptr<OfflineStream> CreateStream() = 0;

  virtual std::unique_ptr<OfflineStream> CreateStream(
      const std::vector<std::vector<int32_t>> &context_list) {
    SHERPA_LOG(FATAL) << "Only transducer models support contextual biasing.";
    return nullptr;  // just to make compiler happy
  }

  virtual void DecodeStreams(OfflineStream **ss, int32_t n) = 0;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_IMPL_H_
