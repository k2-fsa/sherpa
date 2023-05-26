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

dir=$1

# paramters for TRT engines
MIN_BATCH=1
OPT_BATCH=32
MAX_BATCH=128

ENC_MIN_LEN=16
ENC_OPT_LEN=512
ENC_MAX_LEN=1024

trtexec \
--onnx=$dir/encoder.onnx \
--minShapes=speech:${MIN_BATCH}x${ENC_MIN_LEN}x80,speech_lengths:${MIN_BATCH} \
--optShapes=speech:${OPT_BATCH}x${ENC_OPT_LEN}x80,speech_lengths:${OPT_BATCH} \
--maxShapes=speech:${MAX_BATCH}x${ENC_MAX_LEN}x80,speech_lengths:${MAX_BATCH} \
--fp16 \
--saveEngine=$dir/encoder.trt

