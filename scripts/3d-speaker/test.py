#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import argparse

import soundfile as sf
import librosa
import torch
import kaldi_native_fbank as knf
import numpy as np
from typing import Tuple


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    return parser.parse_args()


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(filename: str) -> torch.Tensor:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 2-D float32 tensor of shape (T, 80) containing the features.
    """
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:
        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = []
    opts = knf.FbankOptions()

    opts.frame_opts.dither = 0
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = True

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, wave)
    fbank.input_finished()
    for i in range(fbank.num_frames_ready):
        f = fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)
    # mel (T, 80)

    features = features - features.mean(dim=0, keepdim=True)

    return features


@torch.inference_mode()
def main():
    args = get_args()

    print(f"----------testing {args.model}----------")
    m = torch.jit.load(args.model)
    m.eval()

    x1 = compute_features(filename="./speaker1_a_cn_16k.wav")
    x2 = compute_features(filename="./speaker1_b_cn_16k.wav")
    x3 = compute_features(filename="./speaker2_a_cn_16k.wav")

    y1 = m(x1.unsqueeze(0)).squeeze(0)
    y2 = m(x2.unsqueeze(0)).squeeze(0)
    y3 = m(x3.unsqueeze(0)).squeeze(0)

    print("embedding shape", y1.shape)

    x12 = torch.nn.functional.cosine_similarity(y1, y2, dim=0)
    x13 = torch.nn.functional.cosine_similarity(y1, y3, dim=0)
    x23 = torch.nn.functional.cosine_similarity(y2, y3, dim=0)

    print(x12, x13, x23)
    print(f"----------testing {args.model} done----------")


if __name__ == "__main__":
    main()
