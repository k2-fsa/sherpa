#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

if [ ! -f ./silero_vad_v5.jit ]; then
  # It is silero_vad v5. You can also download it from
  # https://github.com/snakers4/silero-vad/blob/v5.1.2/src/silero_vad/data/silero_vad.jit
  #
  # Note that we have renamed silero_vad.jit to silero_vad_v5.jit
  #
  wget https://huggingface.co/csukuangfj/tmp-files/resolve/main/silero_vad_v5.jit
fi

if [ ! -f ./lei-jun-test.wav ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

if [ ! -f ./Obama.wav ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
fi
