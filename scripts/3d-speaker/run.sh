#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

if [ ! -f ./speaker1_a_cn_16k.wav ]; then
  wget https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/examples/speaker1_a_cn_16k.wav
fi

if [ ! -f ./speaker1_b_cn_16k.wav ]; then
  wget https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/examples/speaker1_b_cn_16k.wav
fi

if [ ! -f ./speaker2_a_cn_16k.wav ]; then
  wget https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/examples/speaker2_a_cn_16k.wav
fi

./export.py

ls -lh

for m in *.pt; do
  ./test.py --model $m
done
