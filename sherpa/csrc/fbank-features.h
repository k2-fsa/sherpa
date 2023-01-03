/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SHERPA_CSRC_FBANK_FEATURES_H_
#define SHERPA_CSRC_FBANK_FEATURES_H_

#include <string>
#include <utility>
#include <vector>

#include "kaldifeat/csrc/feature-fbank.h"
#include "torch/script.h"

namespace sherpa {

/** Read wave samples from a file.
 *
 * If the file has multiple channels, only the first channel is returned.
 * Samples are normalized to the range [-1, 1).
 *
 * @param filename Path to the wave file. Only "*.wav" format is supported.
 * @param expected_sample_rate  Expected sample rate of the wave file. It aborts
 *                              if the sample rate of the given file is not
 *                              equal to this value.
 *
 * @return Return a pair containing
 *  - A 1-D torch.float32 tensor containing entries in the range [-1, 1)
 *  - The duration in seconds of the wave file.
 */
std::pair<torch::Tensor, float> ReadWave(const std::string &filename,
                                         float expected_sample_rate);

/** Compute features for a batch of audio samples in parallel.
 *
 * @param fbank  The Fbank computer.
 * @param wave_data A list of 1-D tensor. Each tensor is of dtype torch.float32
 *                  containing audio samples normalized to the range [-1, 1).
 * @param num_frames If not NULL, on return it will contain the number of
 *                   feature frames for each returned tensor. Though you can
 *                   get the same information after getting the return value,
 *                   it saves computation if you provides it when invoking this
 *                   function.
 * @return It returns the computed features for each input wave data. Each
 *         returned tensor is a 2-D tensor. Its number of rows equals to the
 *         number of feature frames and the number of columns equals to the
 *         feature dimension.
 */
std::vector<torch::Tensor> ComputeFeatures(
    kaldifeat::Fbank &fbank,  // NOLINT
    const std::vector<torch::Tensor> &wave_data,
    std::vector<int64_t> *num_frames = nullptr);
}  // namespace sherpa

#endif  // SHERPA_CSRC_FBANK_FEATURES_H_
