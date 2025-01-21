// sherpa/csrc/voice-activity-detector.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa/csrc/voice-activity-detector.h"

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/voice-activity-detector-impl.h"

namespace sherpa {

void VoiceActivityDetectorConfig::Register(ParseOptions *po) {
  model.Register(po);

  po->Register("vad-segment-size", &segment_size,
               "In seconds. Split input audio into segments and process them "
               "in a batch");

  po->Register("vad-batch-size", &batch_size, "Batch size");
}

bool VoiceActivityDetectorConfig::Validate() const {
  if (segment_size < 0) {
    SHERPA_LOGE("--vad-segment-size='%.3ff' is less than 0", segment_size);
    return false;
  }

  if (batch_size < 1) {
    SHERPA_LOGE("--vad-batch-size='%.3ff' is less than 1", segment_size);
    return false;
  }

  return model.Validate();
}

std::string VoiceActivityDetectorConfig::ToString() const {
  std::ostringstream os;

  os << "VoiceActivityDetectorConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "segment_size=" << segment_size << ", ";
  os << "batch_size=" << batch_size << ")";

  return os.str();
}

VoiceActivityDetector::VoiceActivityDetector(
    const VoiceActivityDetectorConfig &config)
    : impl_(VoiceActivityDetectorImpl::Create(config)) {}

VoiceActivityDetector::~VoiceActivityDetector() = default;

const VoiceActivityDetectorConfig &VoiceActivityDetector::GetConfig() const {
  return impl_->GetConfig();
}

std::vector<SpeechSegment> VoiceActivityDetector::Process(
    torch::Tensor samples) const {
  return impl_->Process(samples);
}

}  // namespace sherpa
