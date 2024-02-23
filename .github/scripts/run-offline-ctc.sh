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
git lfs pull --include "data/lang_bpe_500/HLG.pt"
git lfs pull --include "data/lang_bpe_500/HLG_modified.pt"
popd

log "Decoding with H"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with HLG"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  --hlg=$repo/data/lang_bpe_500/HLG.pt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with HLG (modified H)"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
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


repo_url=https://huggingface.co/csukuangfj/wenet-chinese-model
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "final.zip"
popd

log "Decoding with H"

./build/bin/sherpa-offline \
  --normalize-samples=false \
  --modified=true \
  --nn-model=$repo/final.zip \
  --tokens=$repo/units.txt \
  --use-gpu=false \
  $repo/test_wavs/BAC009S0764W0121.wav \
  $repo/test_wavs/BAC009S0764W0122.wav \
  $repo/test_wavs/BAC009S0764W0123.wav \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav

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

./build/bin/sherpa-offline \
  --nn-model=$repo/wav2vec2_asr_base_10m.pt \
  --tokens=$repo/tokens.txt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with H (voxpopuli_asr_base_10k_de)"

./build/bin/sherpa-offline \
  --nn-model=$repo/voxpopuli_asr_base_10k_de.pt \
  --tokens=$repo/tokens-de.txt \
  --use-gpu=false \
  $repo/test_wavs/20120315-0900-PLENARY-14-de_20120315.wav \
  $repo/test_wavs/20170517-0900-PLENARY-16-de_20170517.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/wgb14/icefall-asr-gigaspeech-conformer-ctc
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe_500/HLG.pt"
git lfs pull --include "data/lang_bpe_500/tokens.txt"

mkdir test_wavs
cd test_wavs
wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1089-134686-0001.wav
wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1221-135766-0001.wav
wget https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio/resolve/main/test_wavs/1221-135766-0002.wav

popd

log "Decoding with H"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decoding with HLG"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  --hlg=$repo/data/lang_bpe_500/HLG.pt \
  --use-gpu=false \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/pkufool/icefall_asr_aishell_conformer_ctc
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_char/HLG.pt"
popd

log "Decoding with H"
./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_char/tokens.txt \
  --use-gpu=false \
  $repo/test_waves/BAC009S0764W0121.wav \
  $repo/test_waves/BAC009S0764W0122.wav \
  $repo/test_waves/BAC009S0764W0123.wav

log "Decoding with HLG"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --hlg=$repo/data/lang_char/HLG.pt \
  --tokens=$repo/data/lang_char/tokens.txt \
  --use-gpu=false \
  $repo/test_waves/BAC009S0764W0121.wav \
  $repo/test_waves/BAC009S0764W0122.wav \
  $repo/test_waves/BAC009S0764W0123.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/AmirHussein/icefall-asr-mgb2-conformer_ctc-2022-27-06
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe_5000/HLG.pt"
git lfs pull --include "data/lang_bpe_5000/tokens.txt"
popd

log "Decoding with H"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --tokens=$repo/data/lang_bpe_5000/tokens.txt \
  --use-gpu=false \
  $repo/test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0053813:0054281.wav \
  $repo/test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0051454:0052244.wav \
  $repo/test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0052244:0053004.wav

log "Decoding with HLG"

./build/bin/sherpa-offline \
  --nn-model=$repo/exp/cpu_jit.pt \
  --hlg=$repo/data/lang_bpe_5000/HLG.pt \
  --tokens=$repo/data/lang_bpe_5000/tokens.txt \
  --use-gpu=false \
  $repo/test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0053813:0054281.wav \
  $repo/test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0051454:0052244.wav \
  $repo/test_wavs/94D37D38-B203-4FC0-9F3A-538F5C174920_spk-0001_seg-0052244:0053004.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/videodanchik/icefall-asr-tedlium3-conformer-ctc2
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo

git lfs pull --include "exp/cpu_jit.pt"

git lfs pull --include "data/lang_bpe/HLG.pt"
git lfs pull --include "data/lang_bpe/tokens.txt"

git lfs pull --include "test_wavs/DanBarber_2010-219.wav"
git lfs pull --include "test_wavs/DanielKahneman_2010-157.wav"
git lfs pull --include "test_wavs/RobertGupta_2010U-15.wav"

popd

log "Decoding with H"
./build/bin/sherpa-offline \
 --nn-model=$repo/exp/cpu_jit.pt \
 --tokens=$repo/data/lang_bpe/tokens.txt \
 $repo/test_wavs/DanBarber_2010-219.wav \
 $repo/test_wavs/DanielKahneman_2010-157.wav \
 $repo/test_wavs/RobertGupta_2010U-15.wav

log "Decoding with HLG"
./build/bin/sherpa-offline \
 --nn-model=$repo/exp/cpu_jit.pt \
 --hlg=$repo/data/lang_bpe/HLG.pt \
 --tokens=$repo/data/lang_bpe/tokens.txt \
 $repo/test_wavs/DanBarber_2010-219.wav \
 $repo/test_wavs/DanielKahneman_2010-157.wav \
 $repo/test_wavs/RobertGupta_2010U-15.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/pkufool/icefall_asr_librispeech_conformer_ctc
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe/HLG.pt"
popd

log "Decoding with H"
./build/bin/sherpa-offline \
 --nn-model=$repo/exp/cpu_jit.pt \
 --tokens=$repo/data/lang_bpe/tokens.txt \
 $repo/test_wavs/1089-134686-0001.wav \
 $repo/test_wavs/1221-135766-0001.wav \
 $repo/test_wavs/1221-135766-0002.wav

log "Decoding with HLG"
./build/bin/sherpa-offline \
 --nn-model=$repo/exp/cpu_jit.pt \
 --hlg=$repo/data/lang_bpe/HLG.pt \
 --tokens=$repo/data/lang_bpe/tokens.txt \
 $repo/test_wavs/1089-134686-0001.wav \
 $repo/test_wavs/1221-135766-0001.wav \
 $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-citrinet-512
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "model.pt"
popd

log "Decoding with H"

./build/bin/sherpa-offline \
  --nn-model=$repo/model.pt \
  --tokens=$repo/tokens.txt \
  --use-gpu=false \
  --modified=false \
  --nemo-normalize=per_feature \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="

repo_url=https://huggingface.co/csukuangfj/sherpa-nemo-ctc-zh-citrinet-512
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "model.pt"
popd

log "Decoding with H"

# the vocab size is huge (e.g., >5000), so we use modified=true here
# to avoid OOM in CI
./build/bin/sherpa-offline \
  --nn-model=$repo/model.pt \
  --tokens=$repo/tokens.txt \
  --use-gpu=false \
  --modified=true \
  --nemo-normalize=per_feature \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav

rm -rf $repo
log "End of testing ${repo_url}"
log "=========================================================================="
