// sherpa/python/csrc/resample.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "sherpa/csrc/resample.h"

#include <algorithm>
#include <memory>

#include "sherpa/python/csrc/resample.h"
#include "sherpa/python/csrc/sherpa.h"
#include "torch/torch.h"

namespace sherpa {

void PybindResample(py::module &m) {  // NOLINT
  using PyClass = LinearResample;
  py::class_<LinearResample>(m, "LinearResample")
      .def(py::init([](int32_t samp_rate_in_hz, int32_t samp_rate_out_hz) {
             float min_freq = std::min(samp_rate_in_hz, samp_rate_out_hz);
             float lowpass_cutoff = 0.99 * 0.5 * min_freq;

             int32_t lowpass_filter_width = 6;
             return std::make_unique<LinearResample>(
                 samp_rate_in_hz, samp_rate_out_hz, lowpass_cutoff,
                 lowpass_filter_width);
           }),
           py::arg("samp_rate_in_hz"), py::arg("samp_rate_out_hz"))
      .def("reset", &PyClass::Reset)
      .def("resample", &PyClass::Resample, py::arg("input"), py::arg("flush"))
      .def_property_readonly("input_sample_rate",
                             &PyClass::GetInputSamplingRate)
      .def_property_readonly("output_sample_rate",
                             &PyClass::GetOutputSamplingRate);
}

}  // namespace sherpa
