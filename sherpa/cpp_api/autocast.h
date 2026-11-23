// sherpa/cpp_api/autocast.h
//
// Copyright (c)  2022  Xiaomi Corporation
#include "ATen/autocast_mode.h"
#include "torch/script.h"

#ifndef SHERPA_CPP_API_AUTO_CAST_H_
#define SHERPA_CPP_API_AUTO_CAST_H_

namespace sherpa {
// This is an RAII class to simulate the context manager torch.autocast()
// from Python.
//
// This class is not intended to be called in a nested environment.
class AutoCast {
 public:
  /**
   * @param use_amp  true to use amp; false to disable amp
   * @param use_gpu  Ignored if use_amp is false.
   *                 true to set amp for CUDA.
   *                 false to set amp for CPU..
   */
  AutoCast(bool use_amp, bool use_gpu) : use_amp_(use_amp), use_gpu_(use_gpu) {
    if (!use_amp_) return;

    if (use_gpu_) {
      at::autocast::set_enabled(true);
    } else {
      at::autocast::set_cpu_enabled(true);
    }
  }
  ~AutoCast() {
    if (!use_amp_) return;

    // by default, the cache for autocast is enabled.
    at::autocast::clear_cache();

    if (use_gpu_) {
      at::autocast::set_enabled(false);
    } else {
      at::autocast::set_cpu_enabled(false);
    }
  }

 private:
  // true to enable amp. false to disable it.
  bool use_amp_;

  // ignored if use_amp_ is false.
  // true to set amp for cuda.
  // false to set amp for cpu.
  bool use_gpu_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_AUTO_CAST_H_
