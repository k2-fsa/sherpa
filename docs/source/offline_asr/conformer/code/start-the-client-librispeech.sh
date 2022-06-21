#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

sherpa/bin/conformer_rnnt/offline_client.py \
  --server-addr localhost \
  --server-port 6010 \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0002.wav

