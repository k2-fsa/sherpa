#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.sample_rates = m.sample_rates
        self.m = m

    @torch.jit.export
    def audio_forward(self, x: torch.Tensor, sr: int, window_size: int = 512):
        # window_size is ignored
        # we wrap v5 so that it has the same interface as v4 for audio_forward
        return self.m.audio_forward(x, sr)


def main():
    m = torch.jit.load("./silero_vad_v5.jit")
    wrapper = Wrapper(m)

    meta_data = {
        "version": "5",
    }
    m = torch.jit.script(wrapper)
    m.save("silero-vad-v5.pt", _extra_files=meta_data)


if __name__ == "__main__":
    main()
