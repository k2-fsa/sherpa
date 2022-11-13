// sherpa/cpp_api/offline-recognizer.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/offline-recognizer.h"

#include <utility>

#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/offline-recognizer-ctc-impl.h"
#include "sherpa/cpp_api/offline-recognizer-impl.h"
#include "sherpa/cpp_api/offline-recognizer-transducer-impl.h"
#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/log.h"
#include "torch/script.h"

namespace sherpa {

void OfflineCtcDecoderConfig::Register(ParseOptions *po) {
  po->Register("modified", &modified,
               "Used only for decoding with a CTC topology. "
               "true to use a modified CTC topology; useful when "
               "vocab_size is large, e.g., > 1000. "
               "false to use a standard CTC topology.");

  po->Register("hlg", &hlg, "Used only for decoding with an HLG graph. ");

  po->Register("search-beam", &search_beam,
               "Used only for CTC decoding. "
               "Decoding beam, e.g. 20.  Smaller is faster, larger is "
               "more exact (less pruning). This is the default value; "
               "it may be modified by `min_active_states` and "
               "`max_active_states`. ");

  po->Register("output-beam", &output_beam,
               "Used only for CTC decoding. "
               "Beam to prune output, similar to lattice-beam in Kaldi. "
               "Relative to best path of output. ");

  po->Register("min-active-states", &min_active_states,
               "Minimum number of FSA states that are allowed to "
               "be active on any given frame for any given "
               "intersection/composition task. This is advisory, "
               "in that it will try not to have fewer than this "
               "number active. Set it to zero if there is no "
               "constraint. ");

  po->Register(
      "max-active-states", &max_active_states,
      "max_activate_states  Maximum number of FSA states that are allowed to "
      "be active on any given frame for any given "
      "intersection/composition task. This is advisory, "
      "in that it will try not to exceed that but may "
      "not always succeed. You can use a very large "
      "number if no constraint is needed. ");
}

void OfflineCtcDecoderConfig::Validate() const {
  if (!hlg.empty()) {
    AssertFileExists(hlg);
  }

  SHERPA_CHECK_GT(search_beam, 0);
  SHERPA_CHECK_GT(output_beam, 0);
  SHERPA_CHECK_GE(min_active_states, 0);
  SHERPA_CHECK_GE(max_active_states, 0);
}

void OfflineRecognizerConfig::Register(ParseOptions *po) {
  ctc_decoder_config.Register(po);
  feat_config.Register(po);

  po->Register("nn-model", &nn_model, "Path to the torchscript model");

  po->Register("tokens", &tokens, "Path to tokens.txt.");

  po->Register("use-gpu", &use_gpu,
               "true to use GPU for computation. false to use CPU.\n"
               "If true, it uses the first device. You can use the environment "
               "variable CUDA_VISIBLE_DEVICES to select which device to use.");

  po->Register("decoding-method", &decoding_method,
               "Decoding method to use. Possible values are: greedy_search, "
               "modified_beam_search");

  po->Register("num-active-paths", &num_active_paths,
               "Number of active paths for modified_beam_search. "
               "Used only when --decoding-method is modified_beam_search");
}

void OfflineRecognizerConfig::Validate() const {
  if (nn_model.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --nn-model";
  }
  AssertFileExists(nn_model);

  if (tokens.empty()) {
    SHERPA_LOG(FATAL) << "Please provide --tokens";
  }
  AssertFileExists(tokens);

  // TODO(fangjun): The following checks about decoding_method are
  // used only for transducer models. We should skip it for CTC models
  if (decoding_method != "greedy_search" &&
      decoding_method != "modified_beam_search") {
    SHERPA_LOG(FATAL)
        << "Unsupported decoding method: " << decoding_method
        << ". Supported values are: greedy_search, modified_beam_search";
  }

  if (decoding_method == "modified_beam_search") {
    SHERPA_CHECK_GT(num_active_paths, 0);
  }
}

std::string OfflineRecognizerConfig::ToString() const {
  // TODO(fangjun): Also print ctc_decoder_config
  std::ostringstream os;
  os << feat_config.ToString() << "\n";
  os << "--nn-model=" << nn_model << "\n";
  os << "--tokens=" << tokens << "\n";
  os << "--use-gpu=" << std::boolalpha << use_gpu << "\n";
  os << "--decoding-method=" << decoding_method << "\n";
  os << "--num-active-paths=" << num_active_paths << "\n";
  return os.str();
}

std::ostream &operator<<(std::ostream &os,
                         const OfflineRecognizerConfig &config) {
  os << config.ToString();
  return os;
}

OfflineRecognizer::~OfflineRecognizer() = default;

OfflineRecognizer::OfflineRecognizer(const OfflineRecognizerConfig &config) {
  if (!config.nn_model.empty()) {
    torch::jit::Module m = torch::jit::load(config.nn_model, torch::kCPU);
    if (!m.hasattr("joiner")) {
      // CTC models do not have a joint network
      impl_ = std::make_unique<OfflineRecognizerCtcImpl>(config);
      return;
    }
  }

  // default to transducer
  impl_ = std::make_unique<OfflineRecognizerTransducerImpl>(config);
}

std::unique_ptr<OfflineStream> OfflineRecognizer::CreateStream() {
  return impl_->CreateStream();
}

void OfflineRecognizer::DecodeStreams(OfflineStream **ss, int32_t n) {
  impl_->DecodeStreams(ss, n);
}

}  // namespace sherpa
