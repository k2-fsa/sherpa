#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

MODEL_NAME="ChatTTS Vocoder"

TRTEXEC="/usr/src/tensorrt/bin/trtexec"

ONNX_PATH="./vocos_vocoder.onnx"
ENGINE_PATH="./vocos_vocoder.plan"

# PRECISION=$1
PRECISION="fp32"
# MAX_BATCH_SIZE=$2
MAX_BATCH_SIZE=8

OPT_BATCH_SIZE=$((MAX_BATCH_SIZE / 2))

MIN_BATCH_SIZE=1

MIN_INPUT_LENGTH=1
OPT_INPUT_LENGTH=1000
MAX_INPUT_LENGTH=2000

MEL_MIN_SHAPE="${MIN_BATCH_SIZE}x100x${MIN_INPUT_LENGTH}"
MEL_OPT_SHAPE="${OPT_BATCH_SIZE}x100x${OPT_INPUT_LENGTH}"
MEL_MAX_SHAPE="${MAX_BATCH_SIZE}x100x${MAX_INPUT_LENGTH}"

MIN_SHAPES="mel:${MEL_MIN_SHAPE}"
OPT_SHAPES="mel:${MEL_OPT_SHAPE}"
MAX_SHAPES="mel:${MEL_MAX_SHAPE}"

echo "Start to build ${MODEL_NAME} engine in ${PRECISION}"

if [ "${PRECISION}" != "fp32" ]; then
    ${TRTEXEC} \
        --minShapes=${MIN_SHAPES} \
        --optShapes=${OPT_SHAPES} \
        --maxShapes=${MAX_SHAPES} \
        --onnx=${ONNX_PATH} \
        --${PRECISION} \
        --saveEngine=${ENGINE_PATH} \
        --verbose=true
else
    ${TRTEXEC} \
        --minShapes=${MIN_SHAPES} \
        --optShapes=${OPT_SHAPES} \
        --maxShapes=${MAX_SHAPES} \
        --onnx=${ONNX_PATH} \
        --saveEngine=${ENGINE_PATH} \
        --verbose=true
fi

die() {
  echo "ERROR: ${@}" 1>&2
  exit 1
}

ls ${ENGINE_PATH} || die "Failed to build ${MODEL_NAME} TensorRT engine."

# rm -v ${ONNX_PATH}

echo "Successfully built ${MODEL_NAME} engine."