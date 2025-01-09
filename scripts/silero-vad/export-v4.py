#!/usr/bin/env python3

import torch


def main():
    m = torch.jit.load("./silero_vad_v4.jit")
    meta_data = {
        "version": "4",
    }
    m.save("silero-vad-v4.pt", _extra_files=meta_data)


if __name__ == "__main__":
    main()
