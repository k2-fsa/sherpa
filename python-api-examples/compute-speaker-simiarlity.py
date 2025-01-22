#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

"""
Please download model files from
https://github.com/k2-fsa/sherpa/releases/

E.g.

wget https://github.com/k2-fsa/sherpa/releases/download/speaker-recognition-models/3d_speaker-speech_eres2netv2_sv_zh-cn_16k-common.pt

Please download test files from
https://github.com/csukuangfj/sr-data/tree/main/test/3d-speaker

"""

import time
from typing import Tuple
import torch

import librosa
import numpy as np
import soundfile as sf

import sherpa


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def create_extractor():
    config = sherpa.SpeakerEmbeddingExtractorConfig(
        model="./3d_speaker-speech_eres2netv2_sv_zh-cn_16k-common.pt",
    )
    print(config)
    return sherpa.SpeakerEmbeddingExtractor(config)


def main():
    extractor = create_extractor()

    file1 = "./speaker1_a_cn_16k.wav"
    file2 = "./speaker1_b_cn_16k.wav"
    file3 = "./speaker2_a_cn_16k.wav"

    samples1, sample_rate1 = load_audio(file1)
    if sample_rate1 != 16000:
        samples1 = librosa.resample(samples1, orig_sr=sample_rate1, target_sr=16000)
        sample_rate1 = 16000

    samples2, sample_rate2 = load_audio(file2)
    if sample_rate2 != 16000:
        samples2 = librosa.resample(samples2, orig_sr=sample_rate2, target_sr=16000)
        sample_rate2 = 16000

    samples3, sample_rate3 = load_audio(file3)
    if sample_rate3 != 16000:
        samples3 = librosa.resample(samples3, orig_sr=sample_rate3, target_sr=16000)
        sample_rate3 = 16000

    start = time.time()
    stream1 = extractor.create_stream()
    stream2 = extractor.create_stream()
    stream3 = extractor.create_stream()

    stream1.accept_waveform(samples1)
    stream2.accept_waveform(samples2)
    stream3.accept_waveform(samples3)

    embeddings = extractor.compute([stream1, stream2, stream3])
    # embeddings: (batch_size, dim)

    x12 = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    x13 = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2], dim=0)
    x23 = torch.nn.functional.cosine_similarity(embeddings[1], embeddings[2], dim=0)

    end = time.time()

    elapsed_seconds = end - start

    print(x12, x13, x23)

    audio_duration = (
        len(samples1) / sample_rate1
        + len(samples2) / sample_rate2
        + len(samples3) / sample_rate3
    )
    real_time_factor = elapsed_seconds / audio_duration
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
