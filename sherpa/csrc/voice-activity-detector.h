// sherpa/csrc/voice-activity-detector.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_H_
#define SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa/cpp_api/parse-options.h"
#include "sherpa/csrc/vad-model-config.h"
#include "torch/script.h"

namespace sherpa {

struct SpeechSegment {
  float start;  // seconds
  float end;    // seconds
};

struct VoiceActivityDetectorConfig {
  VadModelConfig model;
  float segment_size = 10;  // seconds
  int32_t batch_size = 2;

  VoiceActivityDetectorConfig() = default;
  VoiceActivityDetectorConfig(const VadModelConfig &model, float segment_size,
                              int32_t batch_size)
      : model(model), segment_size(segment_size), batch_size(batch_size) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class VoiceActivityDetectorImpl;

class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const VoiceActivityDetectorConfig &config);
  ~VoiceActivityDetector();

  const VoiceActivityDetectorConfig &GetConfig() const;

  /*
   * @param samples 1-D float32 tensor.
   */
  std::vector<SpeechSegment> Process(torch::Tensor samples) const;

 private:
  std::unique_ptr<VoiceActivityDetectorImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_H_
