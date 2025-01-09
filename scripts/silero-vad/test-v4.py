#!/usr/bin/env python3

import torch
import numpy as np
import soundfile as sf
import librosa


def load_audio(filename: str) -> np.ndarray:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)

    if sample_rate != 16000:
        samples = librosa.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=16000,
        )

    return samples


@torch.inference_mode()
def main():
    m = torch.jit.load("./silero-vad-v4.pt")
    m.eval()

    samples = load_audio("./lei-jun-test.wav")
    print(samples.shape)

    batch_size = 1
    h = torch.zeros(2, batch_size, 64, dtype=torch.float32)
    c = torch.zeros(2, batch_size, 64, dtype=torch.float32)
    print(h.shape, c.shape)

    sample_rate = 16000

    start = 0
    window_size = 512
    out = m.audio_forward(
        torch.from_numpy(samples), torch.tensor([sample_rate]), window_size
    )
    # out: (batch_size, num_frames)
    assert out.shape[0] == batch_size, out.shape
    threshold = 0.5
    out = out > threshold
    min_speech_duration = 0.25 * sample_rate / window_size
    min_silence_duration = 0.25 * sample_rate / window_size
    print("min_speech_duration", min_speech_duration)
    for i in range(batch_size):
        w = out[i].tolist()

        result = []
        last = -1
        for k, f in enumerate(w):
            if f >= threshold:
                if last == -1:
                    last = k
            elif last != -1:
                if k - last > min_speech_duration:
                    result.append((last, k))
                last = -1

        if last != -1 and k - last > min_speech_duration:
            result.append((last, k))

        if not result:
            continue
        final = [result[0]]
        for r in result[1:]:
            f = final[-1]
            if r[0] - f[1] < min_silence_duration:
                final[-1] = (f[0], r[1])
            else:
                final.append(r)

        for f in final:
            start = f[0] * window_size / sample_rate
            end = f[1] * window_size / sample_rate
            print("{:.3f} -- {:.3f}".format(start, end))


if __name__ == "__main__":
    main()
