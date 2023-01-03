#!/usr/bin/env bash
cat > wav_streaming.scp <<EOF
wav1 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/test_wavs/1089-134686-0001.wav
wav2 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/test_wavs/1221-135766-0001.wav
wav3 icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/test_wavs/1221-135766-0002.wav
EOF
