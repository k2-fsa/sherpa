#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
from pathlib import Path


speakers = [
    "af_maple",
    "af_sol",
    "bf_vale",
]

d = Path(__file__).parent
for i in range(1, 99 + 1):
    name = "zf_{:03d}".format(i)
    if Path(f"{d}/voices/{name}.txt").is_file():
        speakers.append(name)

for i in range(9, 100 + 1):
    name = "zm_{:03d}".format(i)
    if Path(f"{d}/voices/{name}.txt").is_file():
        speakers.append(name)


id2speaker = {index: value for index, value in enumerate(speakers)}

speaker2id = {speaker: idx for idx, speaker in id2speaker.items()}
