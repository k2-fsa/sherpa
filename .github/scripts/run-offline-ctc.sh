#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe_500/tokens.txt"
git lfs pull --include "data/lang_bpe_500/words.txt"
git lfs pull --include "data/lang_bpe_500/HLG.pt"
git lfs pull --include "data/lang_bpe_500/HLG_modified.pt"
popd

log "Decoding with H"

./build/bin/sherpa-offline \
  --vocab-size=500 \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with HLG"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/words.txt \
  --hlg=$repo/data/lang_bpe_500/HLG.pt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with HLG (modified H)"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/words.txt \
  --hlg=$repo/data/lang_bpe_500/HLG_modified.pt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/wenet-english-model
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "final.zip"
popd

log "Decoding with H"

./build/bin/sherpa-offline \
  --vocab-size=4999 \
  --normalize-samples=false \
  --modified=true \
  --nn-model=$repo/final.zip \
  --tokens=$repo/units.txt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "wav2vec2_asr_base_10m.pt"
git lfs pull --include "voxpopuli_asr_base_10k_de.pt"
popd

log "Decoding with H"

./bin/sherpa-offline \
  --vocab-size=29 \
  --nn-model=$repo/wav2vec2_asr_base_10m.pt \
  --tokens=$repo/tokens.txt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with H (voxpopuli_asr_base_10k_de)"

./bin/sherpa-offline \
  --vocab-size=29 \
  --nn-model=$repo/voxpopuli_asr_base_10k_de.pt \
  --tokens=$repo/tokens-de.txt \
  --use-gpu=false \
  $repo/test_wavs/20120315-0900-PLENARY-14-de_20120315.wav \
  $repo/test_wavs/20170517-0900-PLENARY-16-de_20170517.wav
