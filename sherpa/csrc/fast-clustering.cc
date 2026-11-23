// sherpa/csrc/fast-clustering.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa/csrc/fast-clustering.h"

#include <vector>

#include "torch/torch.h"
//
#include "fastcluster-all-in-one.h"  // NOLINT

namespace sherpa {

class FastClustering::Impl {
 public:
  explicit Impl(const FastClusteringConfig &config) : config_(config) {}

  std::vector<int32_t> Cluster(float *features, int32_t num_rows,
                               int32_t num_cols) const {
    if (num_rows <= 0) {
      return {};
    }

    if (num_rows == 1) {
      return {0};
    }

    torch::Tensor t =
        torch::from_blob(features, {num_rows, num_cols}, torch::kFloat);

    // torch::nn::functional::normalize(
    //     t, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    t.div_(t.norm(2, 1, true));

    std::vector<double> distance((num_rows * (num_rows - 1)) / 2);

    int32_t k = 0;
    for (int32_t i = 0; i != num_rows; ++i) {
      auto v = t.index({i});
      for (int32_t j = i + 1; j != num_rows; ++j) {
        double cosine_similarity = v.dot(t.index({j})).item().toDouble();
        double consine_dissimilarity = 1 - cosine_similarity;

        if (consine_dissimilarity < 0) {
          consine_dissimilarity = 0;
        }

        distance[k] = consine_dissimilarity;
        ++k;
      }
    }

    std::vector<int32_t> merge(2 * (num_rows - 1));
    std::vector<double> height(num_rows - 1);

    fastclustercpp::hclust_fast(num_rows, distance.data(),
                                fastclustercpp::HCLUST_METHOD_COMPLETE,
                                merge.data(), height.data());

    std::vector<int32_t> labels(num_rows);
    if (config_.num_clusters > 0) {
      fastclustercpp::cutree_k(num_rows, merge.data(), config_.num_clusters,
                               labels.data());
    } else {
      fastclustercpp::cutree_cdist(num_rows, merge.data(), height.data(),
                                   config_.threshold, labels.data());
    }

    return labels;
  }

 private:
  FastClusteringConfig config_;
};

FastClustering::FastClustering(const FastClusteringConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FastClustering::~FastClustering() = default;

std::vector<int32_t> FastClustering::Cluster(float *features, int32_t num_rows,
                                             int32_t num_cols) const {
  return impl_->Cluster(features, num_rows, num_cols);
}
}  // namespace sherpa
