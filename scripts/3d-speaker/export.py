#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import os
import pathlib
import re

import torch
from modelscope.hub.snapshot_download import snapshot_download
from speakerlab.bin.infer_sv import supports
from speakerlab.utils.builder import dynamic_import


def convert(model_id):
    local_model_dir = "pretrained"
    save_dir = os.path.join(local_model_dir, model_id.split("/")[1])
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    conf = supports[model_id]
    # download models from modelscope according to model_id
    cache_dir = snapshot_download(
        model_id,
        revision=conf["revision"],
    )
    cache_dir = pathlib.Path(cache_dir)

    download_files = ["examples", conf["model_pt"]]
    for src in cache_dir.glob("*"):
        if re.search("|".join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)

    pretrained_model = save_dir / conf["model_pt"]
    pretrained_state = torch.load(pretrained_model, map_location="cpu")

    model = conf["model"]
    embedding_model = dynamic_import(model["obj"])(**model["args"])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.to("cpu")
    embedding_model.eval()

    x = torch.randn(1, 345, 80)

    m = torch.jit.trace(embedding_model, x)

    meta_data = {
        "model_type": "3d-speaker",
        "version": "1",
        "model_id": model_id,
    }
    m.save(f"3d_speaker-{model_id.split('/')[-1]}.pt", _extra_files=meta_data)
    print(meta_data)


def main():
    for model_id in supports:
        print(f"----------{model_id}----------")
        convert(model_id)


if __name__ == "__main__":
    main()
