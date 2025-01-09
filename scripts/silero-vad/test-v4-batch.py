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

    filenames = ["./lei-jun-test.wav", "./Obama.wav"]

    samples1 = load_audio(filenames[0])
    samples2 = load_audio(filenames[1])
    print(samples1.shape)
    print(samples2.shape)

    samples = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(samples1), torch.from_numpy(samples2)],
        batch_first=True,
        padding_value=0,
    )
    print(samples.shape)

    sample_rate = 16000

    start = 0
    window_size = 512
    out = m.audio_forward(samples, torch.tensor([sample_rate]), window_size)
    # out: (batch_size, num_frames)
    assert out.shape[0] == samples.shape[0], out.shape
    print(out.shape)
    threshold = 0.5
    out = out > threshold
    min_speech_duration = 0.25 * sample_rate / window_size
    min_silence_duration = 0.25 * sample_rate / window_size

    indexes = torch.nonzero(out, as_tuple=False)
    duration = [samples1.shape[0] / sample_rate, samples2.shape[0] / sample_rate]

    for i in range(samples.shape[0]):
        w = indexes[indexes[:, 0] == i, 1].tolist()

        result = []
        start = last = w[0]
        for k in w[1:]:
            if k - last < min_speech_duration:
                last = k
                continue
            else:
                if last - start > min_speech_duration:
                    result.append((start, last))
                start = last = k

        if last - start > min_speech_duration:
            result.append((start, last))

        final = [result[0]]
        for r in result[1:]:
            f = final[-1]
            if r[0] - f[1] < min_silence_duration:
                final[-1] = (f[0], r[1])
            else:
                final.append(r)

        final = filter(lambda f: f[1] - f[0] > min_speech_duration, final)

        print(f"----------{filenames[i]}----------")
        for f in final:
            start = f[0] * window_size / sample_rate
            end = f[1] * window_size / sample_rate
            if start > duration[i] or end > duration[i]:
                break
            print("{:.3f} -- {:.3f}".format(start, end))


if __name__ == "__main__":
    main()
