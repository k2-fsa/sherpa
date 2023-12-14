#!/usr/bin/env python3

import collections
import os
from typing import Any, Dict

import onnx
import torch
from vits import commons, utils
from vits.models import SynthesizerTrn


class OnnxModel(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        x_lengths,
        noise_scale=0.667,
        length_scale=1.0,
        noise_scale_w=0.8,
    ):
        return self.model.infer(
            x=x,
            x_lengths=x_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
        )[0]


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def load_vocab():
    return [
        x.replace("\n", "") for x in open("vocab.txt", encoding="utf-8").readlines()
    ]


@torch.no_grad()
def main():
    hps = utils.get_hparams_from_file("config.json")
    is_uroman = hps.data.training_files.split(".")[-1] == "uroman"
    if is_uroman:
        raise ValueError("We don't support uroman!")

    symbols = load_vocab()

    # Now generate tokens.txt
    all_upper_tokens = [i.upper() for i in symbols]
    duplicate = set(
        [
            item
            for item, count in collections.Counter(all_upper_tokens).items()
            if count > 1
        ]
    )

    print("generate tokens.txt")

    with open("tokens.txt", "w", encoding="utf-8") as f:
        for idx, token in enumerate(symbols):
            f.write(f"{token} {idx}\n")

            # both upper case and lower case correspond to the same ID
            if (
                token.lower() != token.upper()
                and len(token.upper()) == 1
                and token.upper() not in duplicate
            ):
                f.write(f"{token.upper()} {idx}\n")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g.cpu()
    _ = net_g.eval()

    _ = utils.load_checkpoint("G_100000.pth", net_g, None)

    model = OnnxModel(net_g)

    x = torch.randint(low=1, high=10, size=(50,), dtype=torch.int64)
    x = x.unsqueeze(0)

    x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([1], dtype=torch.float32)
    length_scale = torch.tensor([1], dtype=torch.float32)
    noise_scale_w = torch.tensor([1], dtype=torch.float32)

    opset_version = 13

    filename = "model.onnx"

    torch.onnx.export(
        model,
        (x, x_length, noise_scale, length_scale, noise_scale_w),
        filename,
        opset_version=opset_version,
        input_names=[
            "x",
            "x_length",
            "noise_scale",
            "length_scale",
            "noise_scale_w",
        ],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},  # n_audio is also known as batch_size
            "x_length": {0: "N"},
            "y": {0: "N", 2: "L"},
        },
    )
    meta_data = {
        "model_type": "vits",
        "comment": "mms",
        "url": "https://huggingface.co/facebook/mms-tts/tree/main",
        "add_blank": int(hps.data.add_blank),
        "language": os.environ.get("language", "unknown"),
        "frontend": "characters",
        "n_speakers": int(hps.data.n_speakers),
        "sample_rate": hps.data.sampling_rate,
    }
    print("meta_data", meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)


main()
