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
A client for offline ASR recognition.

Usage:
    ./offline_client.py \
      --server-addr localhost \
      --server-port 6006 \
      /path/to/foo.wav \
      /path/to/bar.wav

(Note: You have to first start the server before starting the client)
"""
import argparse
import asyncio

import torchaudio
import websockets


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
        default=6006,
        help="Port of the server",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser.parse_args()


async def main():
    args = get_args()
    assert len(args.sound_files) > 0, f"Empty sound files"

    server_addr = args.server_addr
    server_port = args.server_port

    async with websockets.connect(f"ws://{server_addr}:{server_port}") as websocket:
        for test_wav in args.sound_files:
            print(f"Sending {test_wav}")
            wave, sample_rate = torchaudio.load(test_wav)
            assert sample_rate == 16000, sample_rate

            wave = wave.squeeze(0)
            num_bytes = wave.numel() * wave.element_size()
            await websocket.send((num_bytes).to_bytes(8, "big", signed=True))

            frame_size = (2 ** 20) // 4  # max payload is 1MB
            start = 0
            while start < wave.numel():
                end = start + frame_size
                await websocket.send(wave.numpy().data[start:end])
                start = end
            decoding_results = await websocket.recv()
            print(test_wav, "\n", decoding_results)
            print()


if __name__ == "__main__":
    asyncio.run(main())
