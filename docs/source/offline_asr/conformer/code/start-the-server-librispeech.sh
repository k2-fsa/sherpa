#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

nn_model_filename=./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/cpu_jit-torch-1.6.0.pt
bpe_model=./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/data/lang_bpe_500/bpe.model

sherpa/bin/conformer_rnnt/offline_server.py \
  --port 6010 \
  --num-device 1 \
  --max-batch-size 10 \
  --max-wait-ms 5 \
  --feature-extractor-pool-size 5 \
  --nn-pool-size 2 \
  --max-active-connections 10 \
  --nn-model-filename $nn_model_filename \
  --bpe-model-filename $bpe_model

