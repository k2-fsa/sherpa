#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
onnx_dir=/workspace/icefall/egs/librispeech/ASR/icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/exp
model_repo_dir=/workspace/sherpa/triton/model_repo_streaming
cp $onnx_dir/encoder.onnx $model_repo_dir/encoder/1/
cp $onnx_dir/decoder.onnx $model_repo_dir/decoder/1/
cp $onnx_dir/joiner.onnx $model_repo_dir/joiner/1/

# Start server
tritonserver --model-repository=$model_repo_dir