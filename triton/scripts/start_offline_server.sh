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

jit_model_dir=/workspace/jit_model
model_repo=/workspace/sherpa/triton/model_repo_offline

cp $jit_model_dir/encoder_jit.pt $model_repo/encoder/1
cp $jit_model_dir/decoder_jit.pt $model_repo/decoder/1
cp $jit_model_dir/joiner_jit.pt $model_repo/joiner/1
cp $jit_model_dir/bpe.model /workspace/

# Start server
tritonserver --model-repository=$model_repo


