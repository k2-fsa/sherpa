// sherpa/csrc/test-offline-conformer-transducer-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/offline-conformer-transducer-model.h"
#include "torch/csrc/jit/serialization/pickle.h"
#include "torch/script.h"

// see https://github.com/pytorch/pytorch/issues/20356#issuecomment-1061667333
static std::vector<char> ReadData(const std::string &filename) {
  std::ifstream is(filename, std::ios::binary);
  std::vector<char> ans((std::istreambuf_iterator<char>(is)),
                        (std::istreambuf_iterator<char>()));

  return ans;
}

static void AssertAllClose(torch::Tensor a, torch::Tensor b) {
  if (!torch::allclose(a, b)) {
    std::cerr << "Failed! max abs: " << (a - b).abs().max().item<float>()
              << "\n";
    exit(-1);
  }
}

// Please see ./test-data/test-offline-conformer-transducer-model.py
// for how to generate the test data.
int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr,
            "Usage: ./bin/test-offline-conformer-transducer-model cpu_jit.pt "
            "test_data.pt\n");
    return -1;
  }

  std::string nn_model = argv[1];
  torch::IValue ivalue = torch::jit::pickle_load(ReadData(argv[2]));

  if (!ivalue.isGenericDict()) {
    fprintf(stderr, "Expect a dict.\n");
    return -1;
  }

  auto model =
      std::make_unique<sherpa::OfflineConformerTransducerModel>(nn_model);

  torch::Dict<torch::IValue, torch::IValue> dict = ivalue.toGenericDict();

  torch::Tensor features = dict.at("features").toTensor();
  torch::Tensor features_length = dict.at("features_length").toTensor();

  torch::Tensor encoder_out = dict.at("encoder_out").toTensor();
  torch::Tensor encoder_out_length = dict.at("encoder_out_length").toTensor();

  torch::Tensor decoder_input = dict.at("decoder_input").toTensor();
  torch::Tensor decoder_out = dict.at("decoder_out").toTensor();

  torch::Tensor joiner_out = dict.at("joiner_out").toTensor();

  torch::Tensor encoder_out2;
  torch::Tensor encoder_out2_length;
  std::tie(encoder_out2, encoder_out2_length) =
      model->RunEncoder(features, features_length);

  AssertAllClose(encoder_out, encoder_out2);
  AssertAllClose(encoder_out_length, encoder_out2_length);

  torch::Tensor decoder_out2 = model->RunDecoder(decoder_input);
  AssertAllClose(decoder_out, decoder_out2);

  // see https://pytorch.org/cppdocs/notes/tensor_indexing.html
  using namespace torch::indexing;  // NOLINT

  torch::Tensor joiner_out2 = model->RunJoiner(
      encoder_out2.index({Slice(), Slice(0, 1), Slice()}).unsqueeze(1),
      decoder_out2.unsqueeze(1));

  AssertAllClose(joiner_out, joiner_out2);
  fprintf(stderr, "%s Passed!\n", __FILE__);

  return 0;
}
