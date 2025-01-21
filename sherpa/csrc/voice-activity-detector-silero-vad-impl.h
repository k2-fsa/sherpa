// sherpa/csrc/voice-activity-detector-silero-vad-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_IMPL_SILERO_VAD_H_
#define SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_IMPL_SILERO_VAD_H_

#include <algorithm>

#include "sherpa/csrc/macros.h"
#include "sherpa/csrc/silero-vad-model.h"
#include "sherpa/csrc/voice-activity-detector-impl.h"
#include "torch/torch.h"

namespace sherpa {

std::vector<SpeechSegment> MergeSegments(std::vector<SpeechSegment> segments) {
  std::vector<SpeechSegment> ans;
  ans.reserve(segments.size());
  if (segments.empty()) {
    return ans;
  }

  ans.push_back(std::move(segments[0]));
  for (int32_t i = 1; i < static_cast<int32_t>(segments.size()); ++i) {
    if (ans.back().end + 0.1 >= segments[i].start) {
      ans.back().end = segments[i].end;
    } else {
      ans.push_back(std::move(segments[i]));
    }
  }

  return ans;
}

static std::vector<SpeechSegment> ProcessSegment(
    const float *p, int32_t n, const VoiceActivityDetectorConfig &config,
    int32_t offset, std::vector<SpeechSegment> *out) {
  std::vector<SpeechSegment> ans;
  float threshold = config.model.silero_vad.threshold;
  int32_t temp_start = 0;
  int32_t temp_end = 0;
  bool triggered = false;

  int32_t window_size = 512;
  float sr = 16000.0f;

  float left_shift = 2 * window_size / sr + 0.15;
  float right_shift = 2 * window_size / sr;

  int32_t min_speech_samples = config.model.silero_vad.min_speech_duration *
                               config.model.sample_rate / 512;

  int32_t min_silence_samples = config.model.silero_vad.min_silence_duration *
                                config.model.sample_rate / 512;

  for (int32_t i = 0; i < n; ++i) {
    float prob = p[i];

    if (prob > threshold && temp_end != 0) {
      temp_end = 0;
    }

    if (prob > threshold && temp_start == 0) {
      // start speaking, but we require that it must satisfy
      // min_speech_duration
      temp_start = i;
      continue;
    }

    if (prob > threshold && temp_start != 0 && !triggered) {
      if (i - temp_start < min_speech_samples) {
        continue;
      }
      triggered = true;
      continue;
    }

    if ((prob < threshold) && !triggered) {
      // silence
      temp_start = 0;
      temp_end = 0;
      continue;
    }

    if ((prob > threshold - 0.15) && triggered) {
      // speaking
      continue;
    }

    if ((prob > threshold) && !triggered) {
      // start speaking
      triggered = true;

      continue;
    }

    if ((prob < threshold) && triggered) {
      // stop speaking
      if (temp_end == 0) {
        temp_end = i;
      }

      if (i - temp_end < min_silence_samples) {
        // continue speaking
        continue;
      }
      // stopped speaking

      float start_time = (temp_start + offset) * window_size / sr - left_shift;
      float end_time = (i + offset) * window_size / sr + right_shift;

      start_time = std::max(start_time, 0.0f);

      out->push_back({start_time, end_time});

      temp_start = 0;
      temp_end = 0;
      triggered = false;
    }
  }  // for (int32_t i = 0; i < n; ++i)

  if (triggered) {
    float start_time = (temp_start + offset) * window_size / sr - left_shift;
    float end_time = (n - 1 + offset) * window_size / sr + right_shift;

    start_time = std::max(start_time, 0.0f);

    out->push_back({start_time, end_time});
  }
  return ans;
}

class VoiceActivityDetectorSileroVadImpl : public VoiceActivityDetectorImpl {
 public:
  explicit VoiceActivityDetectorSileroVadImpl(
      const VoiceActivityDetectorConfig &config)
      : config_(config),
        model_(std::make_unique<SileroVadModel>(config.model)) {}

  const VoiceActivityDetectorConfig &GetConfig() const override {
    return config_;
  }

  std::vector<SpeechSegment> Process(torch::Tensor samples) override {
    if (samples.dim() != 1) {
      SHERPA_LOGE("Expect 1-d tensor. Given: %d",
                  static_cast<int32_t>(samples.dim()));
      SHERPA_EXIT(-1);
    }

    int32_t segment_size = config_.model.sample_rate * config_.segment_size;

    int32_t num_samples = samples.size(0);

    bool need_pad =
        (num_samples > segment_size) && (num_samples % segment_size != 0);

    if (need_pad) {
      int32_t padding = segment_size - num_samples % segment_size;
      samples = torch::nn::functional::pad(
          samples, torch::nn::functional::PadFuncOptions({0, padding})
                       .mode(torch::kConstant)
                       .value(0));
    }

    int32_t num_batches = need_pad ? samples.size(0) / segment_size : 1;

    if (need_pad) {
      samples =
          samples.as_strided({num_batches, segment_size}, {segment_size, 1});
    } else {
      samples = samples.reshape({1, -1});
    }

    auto device = model_->Device();
    torch::Tensor probs = model_->Run(samples.to(device)).cpu();
    // probs (batch_size, num_frames)
    int32_t num_frames = probs.size(1);

    std::vector<SpeechSegment> segments;

    for (int32_t i = 0; i < num_batches; ++i) {
      const float *p = probs.data_ptr<float>() + i * num_frames;
      ProcessSegment(p, num_frames, config_, i * num_frames, &segments);
    }

    segments = MergeSegments(std::move(segments));

    return segments;
  }

 private:
  VoiceActivityDetectorConfig config_;
  std::unique_ptr<SileroVadModel> model_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_VOICE_ACTIVITY_DETECTOR_IMPL_SILERO_VAD_H_
