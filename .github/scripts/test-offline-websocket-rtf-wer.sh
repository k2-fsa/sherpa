#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Install lhotse"

python3 -m pip install lhotse websockets

export KALDIFST_MAKE_ARGS="-j4"
log "Install icefall"
git clone http://github.com/k2-fsa/icefall
pushd icefall
pip install -r ./requirements.txt
popd

export PYTHONPATH=$PWD/icefall:$PYTHONPATH

log "Downloading pre-trained model from $repo_url"

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
repo=$(basename $repo_url)

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo

git lfs pull --include "exp/cpu_jit.pt"
ls -lh ./exp/cpu_jit.pt
ls -lh ./data/lang_bpe_500/tokens.txt

popd

log "Downloading test-clean"

wget -q --no-check-certificate https://www.openslr.org/resources/12/test-clean.tar.gz
tar xf test-clean.tar.gz
rm test-clean.tar.gz
ls -lh LibriSpeech

mkdir -p data/manifests
lhotse prepare librispeech -j 2 -p test-clean $PWD/LibriSpeech data/manifests
ls -lh data/manifests

lhotse cut simple \
  -r ./data/manifests/librispeech_recordings_test-clean.jsonl.gz  \
  -s ./data/manifests/librispeech_supervisions_test-clean.jsonl.gz \
  test-clean.jsonl.gz

ls -lh test-clean.jsonl.gz

log "Build sherpa"

mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ENABLE_WEBSOCKET=ON \
  ..

make -j4 sherpa-offline-websocket-server sherpa-offline-websocket-client

ls -lh lib
ls -lh bin

cd ..

log "start the sever"

./build/bin/sherpa-offline-websocket-server \
  --use-gpu=false \
  --port=6006 \
  --num-io-threads=2 \
  --num-work-threads=2 \
  --max-batch-size=5 \
  --nn-model=./$repo/exp/cpu_jit.pt \
  --tokens=./$repo/data/lang_bpe_500/tokens.txt \
  --decoding-method=$DECODING_METHOD \
  --doc-root=./sherpa/bin/web \
  --log-file=./log.txt &

log "Sleep 10 seconds to wait for the server startup"
sleep 10
cat ./log.txt

log "start the client"

# We create 50 concurrent connections here
time python3 ./sherpa/bin/pruned_transducer_statelessX/decode_manifest.py \
  --server-addr 127.0.0.1 \
  --server-port 6006 \
  --manifest-filename ./test-clean.jsonl.gz \
  --num-tasks $NUM_CONNECTIONS
