// sherpa/csrc/offline-speaker-diarization-pyannote-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
#define SHERPA_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa/csrc/fast-clustering.h"
#include "sherpa/csrc/math.h"
#include "sherpa/csrc/offline-speaker-diarization-impl.h"
#include "sherpa/csrc/offline-speaker-segmentation-pyannote-model.h"
#include "sherpa/csrc/speaker-embedding-extractor.h"

namespace sherpa {

namespace {  // NOLINT

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L41
template <class T>
inline void hash_combine(std::size_t *seed, const T &v) {  // NOLINT
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);  // NOLINT
}

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L47
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    std::size_t result = 0;
    hash_combine(&result, pair.first);
    hash_combine(&result, pair.second);
    return result;
  }
};
}  // namespace

using Int32Pair = std::pair<int32_t, int32_t>;

class OfflineSpeakerDiarizationPyannoteImpl
    : public OfflineSpeakerDiarizationImpl {
 public:
  ~OfflineSpeakerDiarizationPyannoteImpl() override = default;

  explicit OfflineSpeakerDiarizationPyannoteImpl(
      const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(config_.segmentation),
        embedding_extractor_(config_.embedding),
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {
    InitPowersetMapping();
    std::cout << "powerset_mapping: " << powerset_mapping_ << "\n";
  }

  int32_t SampleRate() const override {
    const auto &meta_data = segmentation_model_.GetModelMetaData();

    return meta_data.sample_rate;
  }

  void SetConfig(const OfflineSpeakerDiarizationConfig &config) override {
    if (!config.clustering.Validate()) {
      SHERPA_LOGE("Invalid clustering config. Skip it");
      return;
    }
    clustering_ = std::make_unique<FastClustering>(config.clustering);
    config_.clustering = config.clustering;
  }

  OfflineSpeakerDiarizationResult Process(
      torch::Tensor samples,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    std::cout << "samples: " << samples.sizes() << "\n";
    if (samples.dim() != 2) {
      SHERPA_LOGE("Support only 2-d tensors. Given: %d",
                  static_cast<int32_t>(samples.dim()));
      return {};
    }

    if (samples.size(0) != 1) {
      SHERPA_LOGE("Support only batch size == 1. Given: %d",
                  static_cast<int32_t>(samples.size(0)));
      return {};
    }

    std::cout << "samples.sizes: " << samples.sizes() << "\n";
    torch::Tensor log_probs = RunSpeakerSegmentationModel(samples);
    std::cout << "log_probs.sizes: " << log_probs.sizes() << "\n";
    // A chunk is a window_size samples
    // log_probs: (num_chunks, num_frames, 7)
    // where 7 is the num_powerset_classes

    torch::Tensor labels = ToMultiLabel(log_probs);
    std::cout << "labels.sizes: " << labels.sizes() << "\n";

    // labels.sizes: (num_chunks, num_frames, 3)
    // where 3 is num_speakers

    torch::Tensor speakers_per_frame = ComputeSpeakersPerFrame(labels);
    if (speakers_per_frame.argmax().item().toInt() == 0) {
      SHERPA_LOGE("No speakers found");
      return {};
    }
    std::cout << "speakers_per_frame.sizes " << speakers_per_frame.sizes()
              << "\n";

    auto chunk_speaker_samples_list_pair = GetChunkSpeakerSampleIndexes(labels);

    torch::Tensor embeddings =
        ComputeEmbeddings(samples, chunk_speaker_samples_list_pair.second,
                          std::move(callback), callback_arg);
    std::cout << "embedding size: " << embeddings.sizes() << "\n";

    std::vector<int32_t> cluster_labels = clustering_->Cluster(
        embeddings.data_ptr<float>(), embeddings.size(0), embeddings.size(1));

    int32_t max_cluster_index =
        *std::max_element(cluster_labels.begin(), cluster_labels.end());

    auto chunk_speaker_to_cluster = ConvertChunkSpeakerToCluster(
        chunk_speaker_samples_list_pair.first, cluster_labels);

    auto new_labels =
        ReLabel(labels, max_cluster_index, chunk_speaker_to_cluster);
    std::cout << "new_labels.sizes() " << new_labels.sizes() << "\n";

    torch::Tensor speaker_count =
        ComputeSpeakerCount(new_labels, samples.size(1));
    std::cout << "speaker_count.sizes() " << speaker_count.sizes() << "\n";

    torch::Tensor final_labels =
        FinalizeLabels(speaker_count, speakers_per_frame);

    auto result = ComputeResult(final_labels);

    return result;
  }

  torch::Tensor RunSpeakerSegmentationModel(torch::Tensor samples) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;

    int32_t batch_size = samples.size(0);
    int32_t num_samples = samples.size(1);
    int32_t need_pad = (num_samples < window_size) ||
                       ((num_samples - window_size) % window_shift);
    std::cout << "num_samples < window_size: " << (num_samples - window_size)
              << "\n";
    std::cout << "((num_samples - window_size) % window_shift): "
              << ((num_samples - window_size) % window_shift) << "\n";
    std::cout << "need pad: " << need_pad << "\n";

    if (need_pad) {
      int32_t padding = 0;
      if (num_samples < window_size) {
        padding = window_size - num_samples;
      } else {
        padding = window_shift - ((num_samples - window_size) % window_shift);
      }
      std::cout << "padding size: " << padding << "\n";
      samples = torch::nn::functional::pad(
          samples, torch::nn::functional::PadFuncOptions({0, padding, 0, 0})
                       .mode(torch::kConstant)
                       .value(0));
    }
    int32_t num_segments = (samples.size(1) - window_size) / window_shift + 1;

    if (need_pad || num_segments > 1) {
      samples = samples.as_strided({batch_size, num_segments, window_size},
                                   {samples.size(1), window_shift, 1});
    } else {
      samples = samples.reshape({1, 1, -1});
    }

    samples = samples.reshape({-1, 1, window_size});
    // e.g. samples.sizes: (264, 1, 160000)

    int32_t max_batch_size = 2;
    torch::Tensor log_probs;
    if (samples.size(0) < max_batch_size) {
      log_probs = segmentation_model_.Forward(samples);
    } else {
      std::vector<torch::Tensor> tmp;
      int32_t n = samples.size(0) / max_batch_size;
      for (int32_t i = 0; i < n; ++i) {
        auto this_batch =
            samples.slice(0, i * max_batch_size, (i + 1) * max_batch_size);
        std::cout << i << "/" << n << " -> " << this_batch.sizes() << "\n";
        auto this_log_prob = segmentation_model_.Forward(this_batch);
        std::cout << "    " << this_log_prob.sizes() << "\n";
        tmp.push_back(std::move(this_log_prob));
      }

      if (samples.size(0) % max_batch_size) {
        auto this_batch = samples.slice(0, n * max_batch_size);
        std::cout << n << " -> " << this_batch.sizes() << "\n";
        auto this_log_prob = segmentation_model_.Forward(this_batch);
        std::cout << "    " << this_log_prob.sizes() << "\n";
        tmp.push_back(std::move(this_log_prob));
      }

      log_probs = torch::cat(tmp, 0);
    }
    // e.g. log_probs.sizes: (264, 589, 7)
    std::cout << "log_probs.sizes: " << log_probs.sizes() << "\n";

    return log_probs;
  }

  // see
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/utils/powerset.py#L103
  torch::Tensor ToMultiLabel(torch::Tensor log_probs) const {
    int32_t num_classes = powerset_mapping_.size(0);
    auto powerset_probs = torch::nn::functional::one_hot(
                              torch::argmax(log_probs, -1), num_classes)
                              .to(torch::kFloat);
    std::cout << "powerset_probs.sizes: " << powerset_probs.sizes() << "\n";
    auto labels = torch::matmul(powerset_probs, powerset_mapping_);
    std::cout << "labels.sizes: " << labels.sizes() << "\n";
    // labels.size (num_chunks, num_frames, 3)
    return labels;
  }

  // Return a 1-D int32 tensor of shape (num_frames,)
  torch::Tensor ComputeSpeakersPerFrame(torch::Tensor labels) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size(0);

    int32_t num_frames = (window_size + (num_chunks - 1) * window_shift) /
                             receptive_field_shift +
                         1;
    torch::Tensor count = torch::zeros({num_frames}, torch::kFloat);
    torch::Tensor weight = torch::zeros({num_frames}, torch::kFloat);

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start =
          static_cast<float>(i) * window_shift / receptive_field_shift + 0.5;

      int32_t end = start + labels.size(1);

      count.slice(0, start, end).add_(labels.index({i}).sum(1));
      weight.slice(0, start, end).add_(1);
    }

    return (count / (weight + 1e-12f) + 0.5).to(torch::kInt);
  }

  // ans.first: a list of (chunk_id, speaker_id)
  // ans.second: a list of list of (start_sample_index, end_sample_index)
  //
  // ans.first[i] corresponds to ans.second[i]
  std::pair<std::vector<Int32Pair>, std::vector<std::vector<Int32Pair>>>
  GetChunkSpeakerSampleIndexes(torch::Tensor labels) const {
    labels = ExcludeOverlap(labels);
    // now labels.dtype is changed from float to int32

    std::vector<Int32Pair> chunk_speaker_list;
    std::vector<std::vector<Int32Pair>> samples_index_list;

    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;
    int32_t num_speakers = meta_data.num_speakers;

    int32_t num_frames = labels.size(1);
    int32_t num_chunks = labels.size(0);
    for (int32_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
      int32_t sample_offset = chunk_index * window_shift;

      torch::Tensor this_chunk = labels.index({chunk_index}).t();
      // this_chunk: (num_speakers, num_frames)

      for (int32_t speaker_index = 0; speaker_index != num_speakers;
           ++speaker_index) {
        torch::Tensor this_speaker = this_chunk.index({speaker_index});
        if (this_speaker.sum().item().toInt() < 10) {
          // skip segments less than 10 frames
          continue;
        }

        Int32Pair this_chunk_speaker = {chunk_index, speaker_index};
        std::vector<Int32Pair> this_speaker_samples;

        bool is_active = false;
        int32_t start_index = 0;

        auto acc = this_speaker.accessor<int32_t, 1>();

        for (int32_t k = 0; k != num_frames; ++k) {
          if (acc[k] != 0) {
            if (!is_active) {
              is_active = true;
              start_index = k;
            }
          } else if (is_active) {
            is_active = false;

            int32_t start_samples =
                static_cast<float>(start_index) / num_frames * window_size +
                sample_offset;
            int32_t end_samples =
                static_cast<float>(k) / num_frames * window_size +
                sample_offset;

            this_speaker_samples.emplace_back(start_samples, end_samples);
          }
        }

        if (is_active) {
          int32_t start_samples =
              static_cast<float>(start_index) / num_frames * window_size +
              sample_offset;
          int32_t end_samples =
              static_cast<float>(num_frames - 1) / num_frames * window_size +
              sample_offset;
          this_speaker_samples.emplace_back(start_samples, end_samples);
        }

        chunk_speaker_list.push_back(std::move(this_chunk_speaker));
        samples_index_list.push_back(std::move(this_speaker_samples));
      }  // for (int32_t speaker_index = 0;
    }  // for (const auto &label : new_labels)

    return {chunk_speaker_list, samples_index_list};
  }

 private:
  void InitPowersetMapping() {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t num_classes = meta_data.num_classes;
    int32_t powerset_max_classes = meta_data.powerset_max_classes;
    int32_t num_speakers = meta_data.num_speakers;

    powerset_mapping_ =
        torch::zeros({num_classes, num_speakers}, torch::kFloat);
    auto acc = powerset_mapping_.accessor<float, 2>();

    int32_t k = 1;
    for (int32_t i = 1; i <= powerset_max_classes; ++i) {
      if (i == 1) {
        for (int32_t j = 0; j != num_speakers; ++j, ++k) {
          acc[k][j] = 1;
        }
      } else if (i == 2) {
        for (int32_t j = 0; j != num_speakers; ++j) {
          for (int32_t m = j + 1; m < num_speakers; ++m, ++k) {
            acc[k][j] = 1;
            acc[k][m] = 1;
          }
        }
      } else {
        SHERPA_LOGE("powerset_max_classes = %d is currently not supported!", i);
        SHERPA_EXIT(-1);
      }
    }
  }

  // If there are multiple speakers at a frame, then this frame is excluded.
  torch::Tensor ExcludeOverlap(torch::Tensor labels) const {
    torch::Tensor labels_copy = labels.to(torch::kInt);

    torch::Tensor indexes = labels.sum(-1) > 1;
    labels_copy.index_put_({indexes}, 0);
    return labels_copy;
  }

  torch::Tensor ComputeEmbeddings(
      torch::Tensor samples,
      const std::vector<std::vector<Int32Pair>> &sample_indexes,
      OfflineSpeakerDiarizationProgressCallback callback,
      void *callback_arg) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t sample_rate = meta_data.sample_rate;
    torch::Tensor ans =
        torch::empty({static_cast<int32_t>(sample_indexes.size()),
                      embedding_extractor_.Dim()},
                     torch::kFloat);

    int32_t n = samples.size(1);
    int32_t k = 0;
    int32_t cur_row_index = 0;
    const float *ptr = samples.data_ptr<float>();
    for (const auto &v : sample_indexes) {
      auto stream = embedding_extractor_.CreateStream();
      std::vector<float> buffer;

      for (const auto &p : v) {
        int32_t end = (p.second <= n) ? p.second : n;
        buffer.insert(buffer.end(), ptr + p.first, ptr + end);
      }

      stream->AcceptSamples(buffer.data(), buffer.size());

      torch::Tensor embedding = embedding_extractor_.Compute(stream.get());
      ans.index_put_({k}, embedding);

      k += 1;

      if (callback) {
        callback(k, ans.size(0), callback_arg);
      }

    }  // for (const auto &v : sample_indexes)

    return ans;
  }

  std::unordered_map<Int32Pair, int32_t, PairHash> ConvertChunkSpeakerToCluster(
      const std::vector<Int32Pair> &chunk_speaker_pair,
      const std::vector<int32_t> &cluster_labels) const {
    std::unordered_map<Int32Pair, int32_t, PairHash> ans;

    int32_t k = 0;
    for (const auto &p : chunk_speaker_pair) {
      ans[p] = cluster_labels[k];
      k += 1;
    }

    return ans;
  }

  torch::Tensor ReLabel(torch::Tensor labels, int32_t max_cluster_index,
                        const std::unordered_map<Int32Pair, int32_t, PairHash>
                            &chunk_speaker_to_cluster) const {
    int32_t num_chunks = labels.size(0);

    torch::Tensor new_labels = torch::empty(
        {num_chunks, labels.size(1), max_cluster_index + 1}, torch::kFloat);

    for (int32_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
      auto this_chunk = labels.index({chunk_index}).t();
      // this_chunk: (num_speakers, num_frames)

      torch::Tensor new_label = torch::zeros(
          {this_chunk.size(1), max_cluster_index + 1}, torch::kFloat);

      auto this_chunk_acc = this_chunk.accessor<float, 2>();
      auto new_label_acc = new_label.accessor<float, 2>();

      for (int32_t speaker_index = 0; speaker_index != this_chunk.size(1);
           ++speaker_index) {
        if (chunk_speaker_to_cluster.count({chunk_index, speaker_index}) == 0) {
          continue;
        }

        int32_t new_speaker_index =
            chunk_speaker_to_cluster.at({chunk_index, speaker_index});

        for (int32_t k = 0; k != this_chunk.size(1); ++k) {
          if (this_chunk_acc[speaker_index][k] == 1) {
            new_label_acc[k][new_speaker_index] = 1;
          }
        }
      }

      // TODO(fangjun): Optimize it. No need to create a new_label variable
      new_labels.index_put_({chunk_index}, new_label);

      chunk_index += 1;
    }

    return new_labels;
  }

  torch::Tensor ComputeSpeakerCount(torch::Tensor labels,
                                    int32_t num_samples) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size(0);

    int32_t num_frames = (window_size + (num_chunks - 1) * window_shift) /
                             receptive_field_shift +
                         1;

    torch::Tensor count =
        torch::zeros({num_frames, labels.size(2)}, torch::kFloat);

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start =
          static_cast<float>(i) * window_shift / receptive_field_shift + 0.5;
      int32_t end = start + labels.size(1);

      count.slice(0, start, end).add_(labels.index({i}));
    }

    bool has_last_chunk = ((num_samples - window_size) % window_shift) > 0;

    if (!has_last_chunk) {
      return count.to(torch::kInt);
    }

    int32_t last_frame = num_samples / receptive_field_shift;
    return count.slice(0, 0, last_frame).to(torch::kInt);
  }

  // count: float, (num_frames, num_spakers)
  // speakers_per_frame: int,  (num_frames,)
  torch::Tensor FinalizeLabels(torch::Tensor count,
                               torch::Tensor speakers_per_frame) const {
    int32_t num_rows = count.size(0);
    int32_t num_cols = count.size(1);

    torch::Tensor ans = torch::zeros({num_rows, num_cols}, torch::kInt);

    auto speaker_acc = speakers_per_frame.accessor<int32_t, 1>();
    auto ans_acc = ans.accessor<int32_t, 2>();

    for (int32_t i = 0; i != num_rows; ++i) {
      int32_t k = speaker_acc[i];
      if (k == 0) {
        continue;
      }
      torch::Tensor values, indexes;
      std::tie(values, indexes) = count.index({i}).topk(k, 0, true, true);

      auto indexes_acc = indexes.accessor<int64_t, 1>();

      for (int32_t m = 0; m < k; ++m) {
        ans_acc[i][indexes_acc[m]] = 1;
      }
    }

    return ans;
  }

  void MergeSegments(
      std::vector<OfflineSpeakerDiarizationSegment> *segments) const {
    float min_duration_off = config_.min_duration_off;
    bool changed = true;
    while (changed) {
      changed = false;
      for (int32_t i = 0; i < static_cast<int32_t>(segments->size()) - 1; ++i) {
        auto s = (*segments)[i].Merge((*segments)[i + 1], min_duration_off);
        if (s) {
          (*segments)[i] = s.value();
          segments->erase(segments->begin() + i + 1);

          changed = true;
          break;
        }
      }
    }
  }

  OfflineSpeakerDiarizationResult ComputeResult(
      torch::Tensor final_labels) const {
    torch::Tensor final_labels_t = final_labels.t();

    int32_t num_speakers = final_labels_t.size(0);
    int32_t num_frames = final_labels_t.size(1);
    auto acc = final_labels_t.accessor<int32_t, 2>();

    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;
    int32_t receptive_field_size = meta_data.receptive_field_size;
    int32_t sample_rate = meta_data.sample_rate;

    float scale = static_cast<float>(receptive_field_shift) / sample_rate;
    float scale_offset = 0.5 * receptive_field_size / sample_rate;

    OfflineSpeakerDiarizationResult ans;

    for (int32_t speaker_index = 0; speaker_index != num_speakers;
         ++speaker_index) {
      std::vector<OfflineSpeakerDiarizationSegment> this_speaker;

      bool is_active = acc[speaker_index][0] > 0;
      int32_t start_index = is_active ? 0 : -1;

      for (int32_t frame_index = 1; frame_index != num_frames; ++frame_index) {
        if (is_active) {
          if (acc[speaker_index][frame_index] == 0) {
            float start_time = start_index * scale + scale_offset;
            float end_time = frame_index * scale + scale_offset;

            OfflineSpeakerDiarizationSegment segment(start_time, end_time,
                                                     speaker_index);
            this_speaker.push_back(segment);

            is_active = false;
          }
        } else if (acc[speaker_index][frame_index] == 1) {
          is_active = true;
          start_index = frame_index;
        }
      }

      if (is_active) {
        float start_time = start_index * scale + scale_offset;
        float end_time = (num_frames - 1) * scale + scale_offset;

        OfflineSpeakerDiarizationSegment segment(start_time, end_time,
                                                 speaker_index);
        this_speaker.push_back(segment);
      }

      // merge segments if the gap between them is less than min_duration_off
      MergeSegments(&this_speaker);

      for (const auto &seg : this_speaker) {
        if (seg.Duration() > config_.min_duration_on) {
          ans.Add(seg);
        }
      }
    }  // for (int32_t speaker_index = 0; speaker_index != num_speakers;

    return ans;
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  OfflineSpeakerSegmentationPyannoteModel segmentation_model_;
  SpeakerEmbeddingExtractor embedding_extractor_;
  std::unique_ptr<FastClustering> clustering_;
  torch::Tensor powerset_mapping_;  // 2-d float tensor
  /*
 0  0  0   // 0
 1  0  0   // 1
 0  1  0   // 2
 0  0  1   // 3
 1  1  0   // 4
 1  0  1   // 5
 0  1  1   // 6
 */
};

}  // namespace sherpa
#endif  // SHERPA_NNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
