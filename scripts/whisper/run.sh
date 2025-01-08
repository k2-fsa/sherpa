#!/usr/bin/env bash

set -ex

if [ -z $name ]; then
  name=tiny.en
fi

python3 ./export.py --model $name
ls -lh model.pt tokens.txt


cat >README.md << EOF
# Introduction

Models in this file are converted from
https://github.com/openai/whisper
using the following script
https://github.com/k2-fsa/sherpa/blob/master/scripts/whisper/run.sh

EOF
