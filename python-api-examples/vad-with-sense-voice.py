#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

"""
Please download sense voice model from
https://github.com/k2-fsa/sherpa/releases/tag/asr-models

E.g.,
wget https://github.com/k2-fsa/sherpa/releases/download/asr-models/sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06.tar.bz2


Please download VAD models from
https://github.com/k2-fsa/sherpa/releases/tag/vad-models

E.g.,
wget https://github.com/k2-fsa/sherpa/releases/download/vad-models/silero-vad-v4.pt
"""
from typing import Tuple

import librosa
import numpy as np
import sherpa
import soundfile as sf
import torch


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def create_recognizer():
    config = sherpa.OfflineRecognizerConfig(
        model=sherpa.OfflineModelConfig(
            sense_voice=sherpa.OfflineSenseVoiceModelConfig(
                model="./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/model.pt",
                use_itn=True,
                language="auto",
            ),
            debug=False,
        ),
        tokens="./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/tokens.txt",
        use_gpu=False,
    )

    # You have to call config.Validate() to make it work!
    config.validate()
    return sherpa.OfflineRecognizer(config)


def create_vad():
    config = sherpa.VoiceActivityDetectorConfig(
        segment_size=20,
        model=sherpa.VadModelConfig(
            silero_vad=sherpa.SileroVadModelConfig(
                model="./silero-vad-v4.pt",
                threshold=0.5,
                min_speech_duration=0.25,
                min_silence_duration=0.5,
            ),
            sample_rate=16000,
        ),
    )
    return sherpa.VoiceActivityDetector(config)


def main():
    vad = create_vad()
    recognizer = create_recognizer()

    test_wave_file = "./lei-jun-test.wav"

    samples, sample_rate = load_audio(test_wave_file)
    if sample_rate != 16000:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    segments = vad.process(torch.from_numpy(samples))
    for s in segments:
        start_sample = int(s.start * sample_rate)
        end_sample = int(s.end * sample_rate)
        stream = recognizer.create_stream()
        stream.accept_waveform(samples[start_sample:end_sample])
        recognizer.decode_stream(stream)
        text = stream.result.text

        print(f"{s.start:.3f} -- {s.end:.3f} {text}")


if __name__ == "__main__":
    main()
