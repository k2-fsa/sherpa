#!/usr/bin/env python3

"""
./export.py

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
./vad.py --model ./model.pt --wav ./lei-jun-test.wav
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import as_strided
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--wav", type=str, required=True, help="Path to test.wav")

    return parser.parse_args()


class TorchModel:
    def __init__(self, filename):

        meta_data = {
            "num_speakers": "",
            "powerset_max_classes": "",
            "num_classes": "",
            "sample_rate": "",
            "window_size": "",
            "receptive_field_size": "",
            "receptive_field_shift": "",
            "model_type": "",
            "version": "",
            "maintainer": "",
        }

        self.model = torch.jit.load("model.pt", _extra_files=meta_data)
        self.model.eval()

        for k in [
            "num_speakers",
            "powerset_max_classes",
            "sample_rate",
            "window_size",
            "receptive_field_size",
            "receptive_field_shift",
            "version",
        ]:
            meta_data[k] = int(meta_data[k].decode())

        meta_data["model_type"] = meta_data["model_type"].decode()
        meta_data["maintainer"] = meta_data["maintainer"].decode()
        self.meta_data = meta_data

        meta = meta_data

        self.window_size = int(meta["window_size"])
        self.sample_rate = int(meta["sample_rate"])
        self.window_shift = int(0.1 * self.window_size)
        self.receptive_field_size = int(meta["receptive_field_size"])
        self.receptive_field_shift = int(meta["receptive_field_shift"])
        self.num_speakers = int(meta["num_speakers"])
        self.powerset_max_classes = int(meta["powerset_max_classes"])
        self.num_classes = int(meta["num_classes"])

    def __call__(self, x):
        """
        Args:
          x: (N, num_samples)
        Returns:
          A tensor of shape (N, num_frames, num_classes)
        """
        x = np.expand_dims(x, axis=1)
        x = torch.from_numpy(x)
        return self.model(x).numpy()


def load_wav(filename, expected_sample_rate) -> np.ndarray:
    audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != expected_sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=expected_sample_rate,
        )
    return audio


def get_powerset_mapping(num_classes, num_speakers, powerset_max_classes):
    mapping = np.zeros((num_classes, num_speakers))

    k = 1
    for i in range(1, powerset_max_classes + 1):
        if i == 1:
            for j in range(0, num_speakers):
                mapping[k, j] = 1
                k += 1
        elif i == 2:
            for j in range(0, num_speakers):
                for m in range(j + 1, num_speakers):
                    mapping[k, j] = 1
                    mapping[k, m] = 1
                    k += 1
        elif i == 3:
            raise RuntimeError("Unsupported")

    return mapping


def to_multi_label(y, mapping):
    """
    Args:
      y: (num_chunks, num_frames, num_classes)
    Returns:
      A tensor of shape (num_chunks, num_frames, num_speakers)
    """
    y = np.argmax(y, axis=-1)
    labels = mapping[y.reshape(-1)].reshape(y.shape[0], y.shape[1], -1)
    return labels


@torch.inference_mode()
def main():
    args = get_args()
    assert Path(args.model).is_file(), args.model
    assert Path(args.wav).is_file(), args.wav

    m = TorchModel(args.model)
    audio = load_wav(args.wav, m.sample_rate)
    # audio: (num_samples,)
    print("audio", audio.shape, audio.min(), audio.max(), audio.sum())

    num = (audio.shape[0] - m.window_size) // m.window_shift + 1

    samples = as_strided(
        audio,
        shape=(num, m.window_size),
        strides=(m.window_shift * audio.strides[0], audio.strides[0]),
    )

    # or use torch.Tensor.unfold
    #  samples = torch.from_numpy(audio).unfold(0, m.window_size, m.window_shift).numpy()

    print(
        "samples",
        samples.shape,
        samples.mean(),
        samples.sum(),
        samples[:3, :3].sum(axis=-1),
    )

    if (
        audio.shape[0] < m.window_size
        or (audio.shape[0] - m.window_size) % m.window_shift > 0
    ):
        has_last_chunk = True
    else:
        has_last_chunk = False

    num_chunks = samples.shape[0]
    batch_size = 32
    output = []
    for i in range(0, num_chunks, batch_size):
        start = i
        end = i + batch_size
        # it's perfectly ok to use end > num_chunks
        y = m(samples[start:end])
        output.append(y)

    if has_last_chunk:
        last_chunk = audio[num_chunks * m.window_shift :]  # noqa
        pad_size = m.window_size - last_chunk.shape[0]
        last_chunk = np.pad(last_chunk, (0, pad_size))
        last_chunk = np.expand_dims(last_chunk, axis=0)
        y = m(last_chunk)
        output.append(y)

    y = np.vstack(output)
    # y: (num_chunks, num_frames, num_classes)

    mapping = get_powerset_mapping(
        num_classes=m.num_classes,
        num_speakers=m.num_speakers,
        powerset_max_classes=m.powerset_max_classes,
    )
    labels = to_multi_label(y, mapping=mapping)
    # labels: (num_chunks, num_frames, num_speakers)

    # binary classification
    labels = np.max(labels, axis=-1)
    # labels: (num_chunk, num_frames)

    num_frames = (
        int(
            (m.window_size + (labels.shape[0] - 1) * m.window_shift)
            / m.receptive_field_shift
        )
        + 1
    )

    count = np.zeros((num_frames,))
    classification = np.zeros((num_frames,))
    weight = np.hamming(labels.shape[1])

    for i in range(labels.shape[0]):
        this_chunk = labels[i]
        start = int(i * m.window_shift / m.receptive_field_shift + 0.5)
        end = start + this_chunk.shape[0]

        classification[start:end] += this_chunk * weight
        count[start:end] += weight

    classification /= np.maximum(count, 1e-12)

    if has_last_chunk:
        stop_frame = int(audio.shape[0] / m.receptive_field_shift)
        classification = classification[:stop_frame]

    classification = classification.tolist()

    onset = 0.5
    offset = 0.5

    is_active = classification[0] > onset
    start = None
    if is_active:
        start = 0

    scale = m.receptive_field_shift / m.sample_rate
    scale_offset = m.receptive_field_size / m.sample_rate * 0.5

    for i in range(len(classification)):
        if is_active:
            if classification[i] < offset:
                print(
                    f"{start*scale + scale_offset:.3f} -- {i*scale + scale_offset:.3f}"
                )
                is_active = False
        else:
            if classification[i] > onset:
                start = i
                is_active = True

    if is_active:
        print(
            f"{start*scale + scale_offset:.3f} -- {(len(classification)-1)*scale + scale_offset:.3f}"
        )


if __name__ == "__main__":
    main()
