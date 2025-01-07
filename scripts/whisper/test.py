#!/usr/bin/env python3

import torch
import whisper


def main():
    m = torch.jit.load("encoder.pt")
    features = torch.rand(1, 80, 3000)
    n_layer_cross_k, n_layer_cross_v = m.run_encoder(features)
    print(n_layer_cross_k.shape)
    print(n_layer_cross_v.shape)


if __name__ == "__main__":
    main()
