#!/usr/bin/env bash

# This file test ALL known offline transducer models

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
popd

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-offline \
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
  time ./build/bin/sherpa-offline \
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
  time ./build/bin/sherpa-offline \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    --use-feats-scp=true \
    scp:feats.scp \
    ark,scp,t:results2-$m.ark,results2-$m.scp

  head results2-$m.scp results2-$m.ark
done

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="


repo_url=https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data (aishell) from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit-epoch-29-avg-5-torch-1.6.0.pt"
cd exp
ln -sv cpu_jit-epoch-29-avg-5-torch-1.6.0.pt cpu_jit.pt
popd

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-offline \
    --decoding-method=greedy_search \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

.github/scripts/generate_wav_scp_aishell.sh

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-offline \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char/tokens.txt \
    --use-wav-scp=true \
    scp:wav_aishell.scp \
    ark,scp,t:results-aishell-$m.ark,results-aishell-$m.scp

  head results-aishell-$m.scp results-aishell-$m.ark
done

.github/scripts/generate_feats_scp.py scp:wav_aishell.scp ark,scp:feats_aishell.ark,feats_aishell.scp

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-offline \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_char/tokens.txt \
    --use-feats-scp=true \
    scp:feats_aishell.scp \
    ark,scp,t:results-aishell2-$m.ark,results-aishell2-$m.scp

  head results-aishell2-$m.scp results-aishell2-$m.ark
done

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit-torch-1.10.0.pt"
cd exp
ln -s cpu_jit-torch-1.10.0.pt cpu_jit.pt
popd

for m in greedy_search modified_beam_search; do
  time ./build/bin/sherpa-offline \
    --decoding-method=$m \
    --nn-model=$repo/exp/cpu_jit.pt \
    --tokens=$repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="
