#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
popd

log "Test C++ APIs test_decode_files, test_decode_samples, and test_decode_features"

for exe in test_decode_files test_decode_samples test_decode_features; do
  time ./build/bin/$exe \
    $repo/exp/cpu_jit.pt \
    $repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav

  time ./build/bin/$exe \
    $repo/exp/cpu_jit.pt \
    $repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav

  time ./build/bin/$exe \
    $repo/exp/cpu_jit.pt \
    $repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

log "Test ./bin/sherpa"
for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav
done

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

log "Test decoding wav.scp"

.github/scripts/generate_wav_scp.sh

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    --use-wav-scp=true \
    scp:wav.scp \
    ark,scp,t:results-$m.ark,results-$m.scp

  head results-$m.scp results-$m.ark
done

log "Test decoding feats.scp"

export PYTHONPATH=$HOME/tmp/kaldifeat/build/lib:$HOME/tmp/kaldifeat/kaldifeat/python:$PYTHONPATH

.github/scripts/generate_feats_scp.py scp:wav.scp ark,scp:feats.ark,feats.scp

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    --use-feats-scp=true \
    scp:feats.scp \
    ark,scp,t:results2-$m.ark,results2-$m.scp

  head results2-$m.scp results2-$m.ark
done

rm -rfv $repo

repo_url=https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20
repo=$(basename $repo_url)
log "Download pretrained model and test-data (aishell) from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit-epoch-29-avg-5-torch-1.6.0.pt"
cd exp
ln -sv cpu_jit-epoch-29-avg-5-torch-1.6.0.pt cpu_jit.pt
popd

log "Test C++ APIs test_decode_files and test_decode_features (aishell)"

for exe in test_decode_files test_decode_samples test_decode_features; do
  time ./build/bin/$exe \
    $repo/exp/cpu_jit.pt \
    $repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav

  time ./build/bin/$exe \
    $repo/exp/cpu_jit.pt \
    $repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav

  time ./build/bin/$exe \
    $repo/exp/cpu_jit.pt \
    $repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

log "Test ./bin/sherpa (aishell)"

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa \
    --decoding-method=greedy_search \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav
done

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa \
    --decoding-method=greedy_search \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

log "Test decoding wav.scp (aishell)"

.github/scripts/generate_wav_scp_aishell.sh

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char/tokens.txt \
    --use-wav-scp=true \
    scp:wav_aishell.scp \
    ark,scp,t:results-aishell-$m.ark,results-aishell-$m.scp

  head results-aishell-$m.scp results-aishell-$m.ark
done

log "Test decoding feats.scp (aishell)"

.github/scripts/generate_feats_scp.py scp:wav_aishell.scp ark,scp:feats_aishell.ark,feats_aishell.scp

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char/tokens.txt \
    --use-feats-scp=true \
    scp:feats_aishell.scp \
    ark,scp,t:results-aishell2-$m.ark,results-aishell2-$m.scp

  head results-aishell2-$m.scp results-aishell2-$m.ark
done

rm -rfv $repo

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt"
cd exp
ln -sv cpu-jit-epoch-30-avg-10-torch-1.10.0.pt cpu_jit.pt
popd

log "Test ./bin/sherpa-online (conv-emformer)"
for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav
done

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done


log "Test decoding wav.scp (conv-emformer) "

.github/scripts/generate_wav_scp_streaming.sh
for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=greedy_search \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    --use-wav-scp=true \
    scp:wav_streaming.scp \
    ark,scp,t:results-streaming-$m.ark,results-streaming-$m.scp

  head results-streaming-$m.scp results-streaming-$m.ark
done

log "Test Streaming C++ API"

time ./build/bin/test_online_recognizer \
  ./$repo/exp/cpu_jit.pt \
  $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav

time ./build/bin/test_online_recognizer \
  $repo/exp/cpu_jit.pt \
  $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rfv $repo

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/encoder_jit_trace-iter-468000-avg-16.pt"
git lfs pull --include "exp/decoder_jit_trace-iter-468000-avg-16.pt"
git lfs pull --include "exp/joiner_jit_trace-iter-468000-avg-16.pt"

cd exp
ln -sv encoder_jit_trace-iter-468000-avg-16.pt encoder_jit_trace.pt
ln -sv decoder_jit_trace-iter-468000-avg-16.pt decoder_jit_trace.pt
ln -sv joiner_jit_trace-iter-468000-avg-16.pt joiner_jit_trace.pt
popd

time ./build/bin/test_online_recognizer \
  $repo/exp/encoder_jit_trace.pt \
  $repo/exp/decoder_jit_trace.pt \
  $repo/exp/joiner_jit_trace.pt \
  $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav

time ./build/bin/test_online_recognizer \
  $repo/exp/encoder_jit_trace.pt \
  $repo/exp/decoder_jit_trace.pt \
  $repo/exp/joiner_jit_trace.pt \
  $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --encoder-model=$repo/exp/encoder_jit_trace.pt \
    --decoder-model=$repo/exp/decoder_jit_trace.pt \
    --joiner-model=$repo/exp/joiner_jit_trace.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav
done

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --encoder-model=$repo/exp/encoder_jit_trace.pt \
    --decoder-model=$repo/exp/decoder_jit_trace.pt \
    --joiner-model=$repo/exp/joiner_jit_trace.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done
