#!/usr/bin/env bash

sherpa/bin/conformer_rnnt/offline_client.py \
  --server-addr localhost \
  --server-port 6010 \
  ./icefall-aishell-pruned-transducer-stateless3-2022-06-20/test_wavs/BAC009S0764W0121.wav \
  ./icefall-aishell-pruned-transducer-stateless3-2022-06-20/test_wavs/BAC009S0764W0122.wav \
  ./icefall-aishell-pruned-transducer-stateless3-2022-06-20/test_wavs/BAC009S0764W0123.wav

