#!/usr/bin/env bash

if [ ! -f ./silero_vad.jit ]; then
  # It is silero_vad v4. You can also download it from
  # https://github.com/snakers4/silero-vad/blob/v4.0/files/silero_vad.jit
  wget https://huggingface.co/csukuangfj/tmp-files/resolve/main/silero_vad.jit
fi

if [ ! -f ./lei-jun-test.wav ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
fi

if [ ! -f ./Obama.wav ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
fi
