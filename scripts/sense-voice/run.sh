#!/usr/bin/env bash

python3 ./export.py

ls -lh tokens.txt model.pt bpe.model

cat >README.md << EOF
# Introduction

Models in this file are converted from
https://www.modelscope.cn/models/iic/SenseVoiceSmall/summary
using the following script
https://github.com/k2-fsa/sherpa/blob/master/scripts/sense-voice/run.sh

EOF
