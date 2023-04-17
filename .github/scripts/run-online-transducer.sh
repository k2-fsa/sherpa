#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "=========================================================================="

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe_500/LG.pt"
popd

for m in greedy_search modified_beam_search fast_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

# For fast_beam_search with LG
time ./build/bin/sherpa-online \
  --decoding-method=fast_beam_search \
  --nn-model=$repo/exp/cpu_jit.pt \
  --lg=$repo/data/lang_bpe_500/LG.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "End of testing ${repo_url}"

log "=========================================================================="
repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt"
git lfs pull --include "data/lang_bpe_500/LG.pt"
cd exp
ln -sv cpu-jit-epoch-30-avg-10-torch-1.10.0.pt cpu_jit.pt
popd

for m in greedy_search modified_beam_search fast_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

# For fast_beam_search with LG

time ./build/bin/sherpa-online \
  --decoding-method=fast_beam_search \
  --nn-model=$repo/exp/cpu_jit.pt \
  --lg=$repo/data/lang_bpe_500/LG.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/encoder_jit_trace-iter-468000-avg-16.pt"
git lfs pull --include "exp/decoder_jit_trace-iter-468000-avg-16.pt"
git lfs pull --include "exp/joiner_jit_trace-iter-468000-avg-16.pt"
git lfs pull --include "data/lang_bpe_500/LG.pt"

cd exp
ln -sv encoder_jit_trace-iter-468000-avg-16.pt encoder_jit_trace.pt
ln -sv decoder_jit_trace-iter-468000-avg-16.pt decoder_jit_trace.pt
ln -sv joiner_jit_trace-iter-468000-avg-16.pt joiner_jit_trace.pt
popd

for m in greedy_search modified_beam_search fast_beam_search; do
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

# For fast_beam_search with LG
time ./build/bin/sherpa-online \
  --decoding-method=fast_beam_search \
  --encoder-model=$repo/exp/encoder_jit_trace.pt \
  --decoder-model=$repo/exp/decoder_jit_trace.pt \
  --joiner-model=$repo/exp/joiner_jit_trace.pt \
  --lg=$repo/data/lang_bpe_500/LG.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit-epoch-39-avg-6-use-averaged-model-1.pt"
git lfs pull --include "data/lang_bpe_500/LG.pt"
cd exp
ln -sv cpu_jit-epoch-39-avg-6-use-averaged-model-1.pt cpu_jit.pt
popd

for m in greedy_search modified_beam_search fast_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

# For fast_beam_search with LG

time ./build/bin/sherpa-online \
  --decoding-method=fast_beam_search \
  --nn-model=$repo/exp/cpu_jit.pt \
  --lg=$repo/data/lang_bpe_500/LG.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

pushd $repo/test_wavs
sox 1089-134686-0001.wav 1.wav pad 5 5
sox 1221-135766-0001.wav 2.wav pad 5 5
sox 1221-135766-0002.wav 3.wav pad 5 5
sox 1.wav 2.wav 3.wav all-in-one.wav
soxi *.wav
ls -lh *.wav
popd

# For Endpoint testing
for m in greedy_search modified_beam_search fast_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    --use-endpoint=true \
    $repo/test_wavs/all-in-one.wav
done

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit-epoch-25-avg-3.pt"
git lfs pull --include "data/lang_bpe_500/LG.pt"
cd exp
ln -sv cpu_jit-epoch-25-avg-3.pt cpu_jit.pt
popd

for m in greedy_search modified_beam_search fast_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_waves/1089-134686-0001.wav \
    $repo/test_waves/1221-135766-0001.wav \
    $repo/test_waves/1221-135766-0002.wav
done

# For fast_beam_search with LG

time ./build/bin/sherpa-online \
  --decoding-method=fast_beam_search \
  --nn-model=$repo/exp/cpu_jit.pt \
  --lg=$repo/data/lang_bpe_500/LG.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  $repo/test_waves/1089-134686-0001.wav \
  $repo/test_waves/1221-135766-0001.wav \
  $repo/test_waves/1221-135766-0002.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit_epoch_7_avg_1_torch.1.7.1.pt"
git lfs pull --include "data/lang_char/LG.pt"
cd exp
ln -sv cpu_jit_epoch_7_avg_1_torch.1.7.1.pt cpu_jit.pt
popd

for m in greedy_search modified_beam_search fast_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char/tokens.txt \
    $repo/test_wavs/DEV_T0000000000.wav \
    $repo/test_wavs/DEV_T0000000001.wav \
    $repo/test_wavs/DEV_T0000000002.wav
done

# For fast_beam_search with LG

time ./build/bin/sherpa-online \
  --decoding-method=fast_beam_search \
  --nn-model=$repo/exp/cpu_jit.pt \
  --lg=$repo/data/lang_char/LG.pt \
  --tokens=$repo/data/lang_char/tokens.txt \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/ptrnull/icefall-asr-conv-emformer-transducer-stateless2-zh
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit-epoch-11-avg-1.pt"
cd exp
ln -sv cpu_jit-epoch-11-avg-1.pt cpu_jit.pt
popd

for m in greedy_search modified_beam_search fast_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char_bpe/tokens.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/3.wav \
    $repo/test_wavs/4.wav
done

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
popd

for m in greedy_search modified_beam_search fast_beam_search; do
  time ./build/bin/sherpa-online \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char_bpe/tokens.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/3.wav \
    $repo/test_wavs/4.wav
done

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="
