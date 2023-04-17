#!/bin/bash
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
onnx_dir=/mnt/samsung-t7/wend/asr/skip_blanks/icefall-asr-librispeech-pruned_transducer_stateless7_ctc_bs-2023-01-29/exp/
model_repo_dir=/ws/triton/zipformer/model_repo_offline_bs
export PYTHONPATH=$PYTHONPATH:/workspace/k2/k2/python/
export PYTHONPATH=$PYTHONPATH:/workspace/k2/build/lib.linux-x86_64-cpython-38/

cp $onnx_dir/encoder.onnx $model_repo_dir/encoder/1/encoder.onnx
cp $onnx_dir/decoder.onnx $model_repo_dir/decoder/1/decoder.onnx
cp $onnx_dir/joiner.onnx $model_repo_dir/joiner/1/joiner.onnx
cp $onnx_dir/joiner_encoder_proj.onnx $model_repo_dir/joiner_encoder_proj/1/joiner_encoder_proj.onnx
cp $onnx_dir/joiner_decoder_proj.onnx $model_repo_dir/joiner_decoder_proj/1/joiner_decoder_proj.onnx
cp $onnx_dir/lconv.onnx $model_repo_dir/lconv/1/lconv.onnx
cp $onnx_dir/ctc_output.onnx $model_repo_dir/ctc_model/1/ctc_output.onnx
cp $onnx_dir/../data/lang_bpe_500/bpe.model /workspace/
# Start server
tritonserver --model-repository=$model_repo_dir

