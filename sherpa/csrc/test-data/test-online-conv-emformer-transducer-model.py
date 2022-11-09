#!/usr/bin/env python3
# Copyright (c)  2022  Xiaomi Corporation
# flake8: noqa

"""
This file generates test data for ../test-online-conv-emformer-transducer-model.cc

Usage:

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
cd icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
git lfs pull --include "exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt"
cd ..

mkdir build
cd build
make -j test-online-conv-emformer-transducer-model

python3 ../sherpa/csrc/test-data/test-online-conv-emformer-transducer-model.py \
  --nn-model ../icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt

./bin/test-online-conv-emformer-transducer-model \
  ../icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt
  ./test-data.pt
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch


# copied from
# https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conv_emformer_transducer_stateless2/emformer.py#L92
def stack_states(
    state_list: List[Tuple[List[List[torch.Tensor]], List[torch.Tensor]]]
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
    """Stack list of emformer states that correspond to separate utterances
    into a single emformer state so that it can be used as an input for
    emformer when those utterances are formed into a batch.

    Note:
      It is the inverse of :func:`unstack_states`.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the emformer model for a single utterance.
        ``states[i]`` is a tuple of 2 elements of i-th utterance.
        ``states[i][0]`` is the attention caches of i-th utterance.
        ``states[i][1]`` is the convolution caches of i-th utterance.
        ``len(states[i][0])`` and ``len(states[i][1])`` both eqaul to number of layers.  # noqa

    Returns:
      A new state corresponding to a batch of utterances.
      See the input argument of :func:`unstack_states` for the meaning
      of the returned tensor.
    """
    batch_size = len(state_list)

    attn_caches = []
    for layer in state_list[0][0]:
        if batch_size > 1:
            # Note: We will stack attn_caches[layer][s][] later to get attn_caches[layer][s]  # noqa
            attn_caches.append([[s] for s in layer])
        else:
            attn_caches.append([s.unsqueeze(1) for s in layer])
    for b, states in enumerate(state_list[1:], 1):
        for li, layer in enumerate(states[0]):
            for si, s in enumerate(layer):
                attn_caches[li][si].append(s)
                if b == batch_size - 1:
                    attn_caches[li][si] = torch.stack(
                        attn_caches[li][si], dim=1
                    )

    conv_caches = []
    for layer in state_list[0][1]:
        if batch_size > 1:
            # Note: We will stack conv_caches[layer][] later to get conv_caches[layer]  # noqa
            conv_caches.append([layer])
        else:
            conv_caches.append(layer.unsqueeze(0))
    for b, states in enumerate(state_list[1:], 1):
        for li, layer in enumerate(states[1]):
            conv_caches[li].append(layer)
            if b == batch_size - 1:
                conv_caches[li] = torch.stack(conv_caches[li], dim=0)

    return [attn_caches, conv_caches]


# You can download the model from
# https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt


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
    chunk_length = model.encoder.chunk_length
    right_context_length = model.encoder.right_context_length

    pad_length = right_context_length + 2 * 4 + 3

    print("chunk_length:", chunk_length)  # 32
    print("right_context_length:", right_context_length)  # 8
    print("pad_length:", pad_length)  # 19

    chunk_size = chunk_length + pad_length
    chunk_shift = chunk_length

    features = torch.rand(2, chunk_size, 80)
    features_length = torch.tensor([chunk_size, chunk_size], dtype=torch.int64)
    init_state = model.encoder.init_states()
    init_states = stack_states([init_state, init_state])

    num_processed_frames = torch.tensor([0, 0], dtype=torch.int32)

    _, _, state = model.encoder.infer(
        features, features_length, num_processed_frames, init_states
    )
    num_processed_frames += chunk_shift
    encoder_out, encoder_out_length, next_state = model.encoder.infer(
        features, features_length, num_processed_frames, state
    )
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
        "state": state,
        "next_state": next_state,
        "num_processed_frames": num_processed_frames,
        "chunk_size": chunk_size,
        "chunk_shift": chunk_shift,
    }
    torch.save(data, "test-data.pt")


if __name__ == "__main__":
    torch.manual_seed(20221107)
    main()
