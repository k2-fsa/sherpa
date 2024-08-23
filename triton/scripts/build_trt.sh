# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# paramters for TRT engines
MIN_BATCH=1
OPT_BATCH=4
MAX_BATCH=$1
onnx_model=$2
trt_model=$3

ENC_MIN_LEN=16
ENC_OPT_LEN=512
ENC_MAX_LEN=2000

/usr/src/tensorrt/bin/trtexec \
--onnx=$onnx_model \
--minShapes=x:${MIN_BATCH}x${ENC_MIN_LEN}x80,x_lens:${MIN_BATCH} \
--optShapes=x:${OPT_BATCH}x${ENC_OPT_LEN}x80,x_lens:${OPT_BATCH} \
--maxShapes=x:${MAX_BATCH}x${ENC_MAX_LEN}x80,x_lens:${MAX_BATCH} \
--fp16 \
--loadInputs=x:scripts/test_features/input_tensor_fp32.dat,x_lens:scripts/test_features/shape.bin \
--shapes=x:1x663x80,x_lens:1 \
--saveEngine=$trt_model

