// sherpa/cpp_api/offline-recognizer.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/cpp_api/offline-recognizer.h"

#include <utility>

#include "sherpa/csrc/file-utils.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/offline-conformer-transducer-model.h"
#include "sherpa/csrc/offline-transducer-decoder.h"
#include "sherpa/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa/csrc/offline-transducer-model.h"
#include "sherpa/csrc/offline-transducer-modified-beam-search-decoder.h"
#include "sherpa/csrc/symbol-table.h"
#include "torch/script.h"

namespace sherpa {

void OfflineRecognizerConfig::Register(ParseOptions *po) {
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

static OfflineRecognitionResult Convert(OfflineTransducerDecoderResult src,
                                        const SymbolTable &sym) {
  OfflineRecognitionResult r;
  std::string text;
  for (auto i : src.tokens) {
    text.append(sym[i]);
  }
  r.text = std::move(text);
  r.tokens = std::move(src.tokens);
  r.timestamps = std::move(src.timestamps);

  return r;
}

class OfflineRecognizer::OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerImpl(const OfflineRecognizerConfig &config)
      : symbol_table_(config.tokens),
        fbank_(config.feat_config.fbank_opts),
        device_(torch::kCPU) {
    if (config.use_gpu) {
      device_ = torch::Device("cuda:0");
    }
    model_ = std::make_unique<OfflineConformerTransducerModel>(config.nn_model,
                                                               device_);

    if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineTransducerGreedySearchDecoder>(model_.get());
    } else if (config.decoding_method == "modified_beam_search") {
      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchDecoder>(
          model_.get(), config.num_active_paths);
    } else {
      TORCH_CHECK(false,
                  "Unsupported decoding method: ", config.decoding_method);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() {
    return std::make_unique<OfflineStream>(&fbank_);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) {
    torch::NoGradGuard no_grad;

    std::vector<torch::Tensor> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      const auto &f = ss[i]->GetFeatures();
      features_vec[i] = f;
      features_length_vec[i] = f.size(0);
    }

    auto features = torch::nn::utils::rnn::pad_sequence(
                        features_vec, /*batch_first*/ true,
                        /*padding_value*/ -23.025850929940457f)
                        .to(device_);

    auto features_length = torch::tensor(features_length_vec).to(device_);

    torch::Tensor encoder_out;
    torch::Tensor encoder_out_length;

    std::tie(encoder_out, encoder_out_length) =
        model_->RunEncoder(features, features_length);
    encoder_out_length = encoder_out_length.cpu();

    auto results = decoder_->Decode(encoder_out, encoder_out_length);
    for (int32_t i = 0; i != n; ++i) {
      ss[i]->SetResult(Convert(results[i], symbol_table_));
    }
  }

 private:
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineTransducerModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
  kaldifeat::Fbank fbank_;
  torch::Device device_;
};

OfflineRecognizer::~OfflineRecognizer() = default;

OfflineRecognizer::OfflineRecognizer(const OfflineRecognizerConfig &config)
    : impl_(std::make_unique<OfflineRecognizerImpl>(config)) {}

std::unique_ptr<OfflineStream> OfflineRecognizer::CreateStream() {
  return impl_->CreateStream();
}

void OfflineRecognizer::DecodeStreams(OfflineStream **ss, int32_t n) {
  impl_->DecodeStreams(ss, n);
}

}  // namespace sherpa
