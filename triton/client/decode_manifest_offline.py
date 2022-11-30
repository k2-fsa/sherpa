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

    ./decode_manifest_offline.py

(Note: You have to first start the server before starting the client)
"""

import argparse
import sys
from functools import partial
import asyncio
import time
from pathlib import Path
import types

import numpy as np
from icefall.utils import store_transcripts, write_error_stats
from lhotse import CutSet, load_manifest

import tritonclient
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype, InferenceServerException

DEFAULT_MANIFEST_FILENAME = "/mnt/samsung-t7/yuekai/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz"  # noqa

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Port of the server",
    )

    parser.add_argument(
        "--manifest-filename",
        type=str,
        default=DEFAULT_MANIFEST_FILENAME,
        help="Path to the manifest for decoding",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="transducer",
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks to use for sending",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Controls how frequently we print the log.",
    )

    parser.add_argument(
        "--compute-cer",
        action="store_true",
        default=False,
        help="""True to compute CER, e.g., for Chinese.
        False to compute WER, e.g., for English words.
        """,
    )

    return parser.parse_args()


async def send(
    cuts: CutSet,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    compute_cer: bool,
    model_name: str,
):
    total_duration = 0.0
    results = []

    for i, c in enumerate(cuts):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(cuts)}")

        waveform = c.load_audio().reshape(-1).astype(np.float32)
        sample_rate = 16000

        # padding to nearset 10 seconds
        samples = np.zeros((1, 10 * sample_rate*(int(len(waveform)/sample_rate // 10) +1)),dtype=np.float32)
        samples[0,:len(waveform)] = waveform

        lengths = np.array([[len(waveform)]], dtype=np.int32)

        inputs = [
            protocol_client.InferInput("WAV", samples.shape,
                                            np_to_triton_dtype(samples.dtype)),
            protocol_client.InferInput("WAV_LENS", lengths.shape,
                                            np_to_triton_dtype(lengths.dtype))
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        sequence_id = 10086 + i

        response = await triton_client.infer(model_name,
                                inputs,
                                request_id=str(sequence_id),
                                outputs=outputs)

        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        decoding_results = b' '.join(decoding_results).decode('utf-8')

        total_duration += c.duration

        if compute_cer:
            ref = c.supervisions[0].text.split()
            hyp = decoding_results.split()
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results.append((c.id, ref, hyp))
        else:
            results.append(
                    (
                        c.id,
                        c.supervisions[0].text.split(),
                        decoding_results.split(),
                    )
                )  # noqa

    return total_duration, results


async def main():
    args = get_args()
    filename = args.manifest_filename
    server_addr = args.server_addr
    server_port = args.server_port
    url = f"{server_addr}:{server_port}"
    num_tasks = args.num_tasks
    log_interval = args.log_interval
    compute_cer = args.compute_cer

    cuts = load_manifest(filename)
    cuts_list = cuts.split(num_tasks)
    tasks = []

    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    protocol_client = grpcclient
    start_time = time.time()
    for i in range(num_tasks):
        task = asyncio.create_task(
            send(
                cuts=cuts_list[i],
                name=f"task-{i}",
                triton_client=triton_client,
                protocol_client=protocol_client,
                log_interval=log_interval,
                compute_cer=compute_cer,
                model_name=args.model_name
            )
        )
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

    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += (
        f"processing time: {elapsed:.3f} seconds "
        f"({elapsed/3600:.2f} hours)\n"
    )
    print(s)

    with open("rtf.txt", "w") as f:
        f.write(s)

    name = Path(filename).stem.split(".")[0]
    results = sorted(results)
    store_transcripts(filename=f"recogs-{name}.txt", texts=results)

    with open(f"errs-{name}.txt", "w") as f:
        write_error_stats(f, "test-set", results, enable_log=True)

    with open(f"errs-{name}.txt", "r") as f:
        print(f.readline())  # WER
        print(f.readline())  # Detailed errors


if __name__ == "__main__":
    asyncio.run(main())
