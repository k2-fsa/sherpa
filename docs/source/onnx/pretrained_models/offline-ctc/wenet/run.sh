#!/usr/bin/env bash

set -ex

python3 ./generate_sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.py
python3 ./generate_sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03.py

ls -lh ./generated/*/
