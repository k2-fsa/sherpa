#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "=========================================================================="
curl -SL -O https://github.com/k2-fsa/sherpa/releases/download/asr-models/sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06.tar.bz2
tar xvf sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06.tar.bz2
rm sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06.tar.bz2
ls -lh sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06

./build/bin/sherpa-offline \
  --debug=1 \
  --sense-voice-model=./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/model.pt \
  --tokens=./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/tokens.txt \
  ./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/test_wavs/en.wav \
  ./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/test_wavs/ja.wav \
  ./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/test_wavs/ko.wav \
  ./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/test_wavs/yue.wav \
  ./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/test_wavs/zh.wav

rm -rf sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06
