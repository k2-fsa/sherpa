#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

nn_model_filename=./icefall-aishell-pruned-transducer-stateless3-2022-06-20/exp/cpu_jit-epoch-29-avg-5-torch-1.6.0.pt
token_filename=./icefall-aishell-pruned-transducer-stateless3-2022-06-20/data/lang_char/tokens.txt

sherpa/bin/conformer_rnnt/offline_server.py \
  --port 6010 \
  --num-device 1 \
  --max-batch-size 10 \
  --max-wait-ms 5 \
  --feature-extractor-pool-size 5 \
  --nn-pool-size 2 \
  --max-active-connections 10 \
  --nn-model-filename $nn_model_filename \
  --token-filename $token_filename

