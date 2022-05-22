#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script loads a manifest in lhotse format and sends it to the server
for decoding, in parallel.

Usage:

    ./decode_mainifest.py

(Note: You have to first start the server before starting the client)
"""

import asyncio
import time

import numpy as np
import websockets
from icefall.utils import store_transcripts, write_error_stats
from lhotse import CutSet, load_manifest


async def send(cuts: CutSet, name: str):
    total_duration = 0.0
    results = []
    async with websockets.connect("ws://localhost:6006") as websocket:
        for i, c in enumerate(cuts):
            if i % 5 == 0:
                print(f"{name}: {i}/{len(cuts)}")

            samples = c.load_audio().reshape(-1).astype(np.float32)
            num_bytes = samples.nbytes

            await websocket.send((num_bytes).to_bytes(8, "big", signed=True))

            frame_size = (2 ** 20) // 4  # max payload is 1MB
            start = 0
            while start < samples.size:
                end = start + frame_size
                await websocket.send(samples.data[start:end])
                start = end
            decoding_results = await websocket.recv()

            total_duration += c.duration

            results.append((c.supervisions[0].text.split(), decoding_results.split()))

    return total_duration, results


async def main():
    filename = "/ceph-fj/fangjun/open-source-2/icefall-master-2/egs/librispeech/ASR/data/fbank/cuts_test-clean.json.gz"
    cuts = load_manifest(filename)
    num_tasks = 50  # we start this number of tasks to send the requests
    cuts_list = cuts.split(num_tasks)
    tasks = []

    start_time = time.time()
    for i in range(num_tasks):
        task = asyncio.create_task(send(cuts_list[i], f"task-{i}"))
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    results = []
    total_duration = 0.0
    for ans in ans_list:
        total_duration += ans[0]
        results += ans[1]

    rtf = elapsed / total_duration

    print(f"RTF: {rtf:.4f}")
    print(
        f"total_duration: {total_duration:.2f} seconds "
        f"({total_duration/3600:.2f} hours)"
    )
    print(f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)")

    store_transcripts(filename="recogs-test-clean.txt", texts=results)
    with open("errs-test-clean.txt", "w") as f:
        wer = write_error_stats(f, "test-set", results, enable_log=True)


if __name__ == "__main__":
    asyncio.run(main())
