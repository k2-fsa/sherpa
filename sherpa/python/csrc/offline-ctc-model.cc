// sherpa/python/csrc/offline-ctc-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/python/csrc/offline-ctc-model.h"

#include <utility>

#include "sherpa/csrc/offline-ctc-model.h"

namespace sherpa {

static constexpr const char *kWarmUpDoc = R"doc(
Send some fake data to the model for computation.

.. hint::

    It is called when the model is first loaded into memory to
    reduce the response time of the first client request.

Args:
  features:
    It is usually a tensor of shape ``(N, T, C)`` containing the features.
    But for ``wav2vec 2.0``, it should be a tensor of shape ``(N, num_samples)``.

  features_length:
    It indicates number of valid frames or audio samples in ``features``
    before padding. Its shape is ``(N,)``.

Returns:
  Return ``None``.
)doc";

static constexpr const char *kForwardDoc = R"doc(
Run the forward method of the network.

Args:
  features:
    It is usually a tensor of shape ``(N, T, C)`` containing the features.
    But for ``wav2vec 2.0``, it should be a tensor of shape ``(N, num_samples)``.

  features_length:
    It indicates number of valid frames or audio samples in ``features``
    before padding. Its shape is ``(N,)``.

Returns:
  Return a tuple containing two tensors:

    - ``log_probs``: Output of the log_softmax layer with shape ``(N, T, vocab_size)``

    - ``log_probs_length``: A tensor of shape ``(N,)`` containing the valid number
      of frames in ``log_probs``
)doc";

void PybindOfflineCtcModel(py::module &m) {  // NOLINT
  using PyClass = OfflineCtcModel;
  py::class_<PyClass>(m, "OfflineCtcModel")
      // properties
      .def_property_readonly("subsampling_factor", &PyClass::SubsamplingFactor)
      .def_property_readonly("vocab_size", &PyClass::VocabSize)
      .def_property_readonly("device",
                             [](const PyClass &self) -> py::object {
                               py::object ans =
                                   py::module_::import("torch").attr("device");
                               return ans(self.Device().str());
                             })
      // methods
      .def("warm_up", &PyClass::WarmUp, py::arg("features"),
           py::arg("features_length"), py::call_guard<py::gil_scoped_release>(),
           kWarmUpDoc)
      .def(
          "forward",
          [](PyClass &self, torch::Tensor features,
             torch::Tensor features_length)
              -> std::pair<torch::Tensor, torch::Tensor> {
            torch::IValue ivalue = self.Forward(features, features_length);

            auto log_probs = self.GetLogSoftmaxOut(ivalue);
            auto log_probs_length = self.GetLogSoftmaxOutLength(ivalue);

            return {log_probs, log_probs_length};
          },
          py::arg("features"), py::arg("features_length"),
          py::call_guard<py::gil_scoped_release>(), kForwardDoc);
}

}  // namespace sherpa
