#!/usr/bin/env python3
# Copyright (c)  2022  Xiaomi Corporation
# flake8: noqa

"""
This file generates test data for ../test-offline-conformer-transducer-model.cc

Usage:

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
cd icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
git lfs pull --include "exp/cpu_jit.pt"
cd ..

mkdir build
cd build
make -j test-offline-conformer-transducer-model

python3 ../sherpa/csrc/test-data/test-offline-conformer-transducer-model.py \
  --nn-model ../icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/cpu_jit.pt

./bin/test-offline-conformer-transducer-model \
  ../icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/cpu_jit.pt \
  ./test-data.pt
"""

import argparse
from pathlib import Path

import torch

# You can download the model from
# https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/blob/main/exp/cpu_jit.pt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nn-model",
        type=str,
        required=True,
        help="Path to the torchscript model",
    )
    return parser.parse_args()


def main():
    args = get_args()
    filename = Path(args.nn_model)
    if not filename.is_file():
        raise ValueError(f"{filename} does not exist")
    model = torch.jit.load(filename)
    features = torch.rand(2, 20, 80)
    features_length = torch.tensor([12, 20], dtype=torch.int64)

    encoder_out, encoder_out_length = model.encoder(features, features_length)
    encoder_out = model.joiner.encoder_proj(encoder_out)

    decoder_input = torch.tensor([[1, 5], [3, 9]], dtype=torch.int64)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    joiner_out = model.joiner(
        encoder_out[:, 0:1, :].unsqueeze(1),
        decoder_out.unsqueeze(1),
        project_input=False,
    )

    print(encoder_out.shape)  # (2, 4, 512)
    print(encoder_out_length)  # [2, 4]
    print(decoder_out.shape)  # (2, 1, 512)
    print(joiner_out.shape)  # (2, 1, 1, 500)
    data = {
        "features": features,
        "features_length": features_length,
        "encoder_out": encoder_out,
        "encoder_out_length": encoder_out_length,
        "decoder_input": decoder_input,
        "decoder_out": decoder_out,
        "joiner_out": joiner_out,
    }
    torch.save(data, "test-data.pt")


if __name__ == "__main__":
    torch.manual_seed(20221106)
    main()
