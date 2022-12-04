#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Download pretrained models"

mkdir -p /tmp/icefall-models
pushd /tmp/icefall-models

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe_500/tokens.txt"
git lfs pull --include "data/lang_bpe_500/HLG.pt"
popd

repo_url=https://huggingface.co/csukuangfj/wenet-english-model
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "final.zip"
popd

repo_url=https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "wav2vec2_asr_base_960h.pt"
popd

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
popd

# Go to /path/to/sherpa
popd

cd sherpa/python/test

pip install pytest

pytest -s -v
