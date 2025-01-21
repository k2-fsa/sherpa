#!/usr/bin/env python3
#
# Copyright (c)  2025  Xiaomi Corporation

"""
Please download sense voice model from
https://github.com/k2-fsa/sherpa/releases/tag/asr-models

E.g.,
wget https://github.com/k2-fsa/sherpa/releases/download/asr-models/sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06.tar.bz2
"""
import librosa
import numpy as np
import sherpa
import soundfile as sf
from typing import Tuple
import time


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


def test_decoding_single_file(recognizer):
    print("----------Test a single file----------")
    test_wave_file = "./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/test_wavs/zh.wav"

    samples, sample_rate = load_audio(test_wave_file)
    if sample_rate != 16000:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    start = time.time()

    stream = recognizer.create_stream()
    stream.accept_waveform(samples)
    recognizer.decode_stream(stream)
    text = stream.result.text

    end = time.time()

    elapsed_seconds = end - start
    audio_duration = len(samples) / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    print(text)
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


def test_decoding_multipl_files(recognizer):
    print("----------Test decoding multiple files----------")
    test_wave_file1 = "./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/test_wavs/zh.wav"
    test_wave_file2 = "./sherpa-sense-voice-zh-en-ja-ko-yue-2025-01-06/test_wavs/en.wav"

    samples1, sample_rate1 = load_audio(test_wave_file1)
    if sample_rate1 != 16000:
        samples1 = librosa.resample(samples1, orig_sr=sample_rate1, target_sr=16000)
        sample_rate1 = 16000

    samples2, sample_rate2 = load_audio(test_wave_file2)
    if sample_rate2 != 16000:
        samples2 = librosa.resample(samples2, orig_sr=sample_rate2, target_sr=16000)
        sample_rate2 = 16000

    start = time.time()
    stream1 = recognizer.create_stream()
    stream1.accept_waveform(samples1)

    stream2 = recognizer.create_stream()
    stream2.accept_waveform(samples2)

    recognizer.decode_streams([stream1, stream2])
    text1 = stream1.result.text
    text2 = stream2.result.text

    end = time.time()

    elapsed_seconds = end - start
    audio_duration = len(samples1) / sample_rate1 + len(samples2) / sample_rate2
    real_time_factor = elapsed_seconds / audio_duration

    print(f"{test_wave_file1}\n  {text1}")
    print()
    print(f"{test_wave_file2}\n  {text2}")

    print()

    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


def main():
    recognizer = create_recognizer()
    test_decoding_single_file(recognizer)
    test_decoding_multipl_files(recognizer)


if __name__ == "__main__":
    main()
