#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "=========================================================================="
model_list=(
base
base.en
distil-large-v2
distil-medium.en
distil-small.en
medium
medium.en
small
small.en
tiny
tiny.en
turbo
)

curl -SL -O https://hf-mirror.com/csukuangfj/sherpa-onnx-zh-wenet-aishell2/resolve/main/test_wavs/0.wav
mv 0.wav zh.wav

for m in ${model_list[@]}; do
  d=sherpa-whisper-$m
  log "----------testing $d----------"
  curl -SL -O https://github.com/k2-fsa/sherpa/releases/download/asr-models/$d.tar.bz2
  tar xvf $d.tar.bz2
  rm $d.tar.bz2
  ls -lh $d

  if [[ $d == *en ]]; then
    log "decode a single file"

    ./build/bin/sherpa-offline \
      --debug=1 \
      --whisper-model=./$d/model.pt \
      --tokens=./$d/tokens.txt \
      ./$d/test_wavs/0.wav

    log "decode two files"
    ./build/bin/sherpa-offline \
      --debug=1 \
      --whisper-model=./$d/model.pt \
      --tokens=./$d/tokens.txt \
      ./$d/test_wavs/0.wav \
      ./$d/test_wavs/1.wav
  fi

  if [[ $d != *en ]]; then

    log "decode a single file"

    ./build/bin/sherpa-offline \
      --debug=1 \
      --whisper-model=./$d/model.pt \
      --tokens=./$d/tokens.txt \
      ./$d/test_wavs/0.wav

    log "decode two files"
    ./build/bin/sherpa-offline \
      --debug=1 \
      --whisper-model=./$d/model.pt \
      --tokens=./$d/tokens.txt \
      ./$d/test_wavs/0.wav \
      ./$d/test_wavs/1.wav

    log "decode three files"
    ./build/bin/sherpa-offline \
      --debug=1 \
      --whisper-model=./$d/model.pt \
      --tokens=./$d/tokens.txt \
      ./$d/test_wavs/0.wav \
      ./$d/test_wavs/1.wav \
      ./zh.wav
  fi
  rm -rf $d
done
