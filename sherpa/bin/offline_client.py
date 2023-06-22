#!/usr/bin/env python3
# Copyright      2022-2023  Xiaomi Corp.
"""
A client for offline ASR.

Usage:
    ./offline_client.py \
      --server-addr localhost \
      --server-port 6006 \
      /path/to/foo.wav \
      /path/to/bar.wav

Note: You have to first start the server before starting the client.
You can use either ./offline_transducer_server.py or ./offline_ctc_server.py
"""
import argparse
import asyncio
import http
import logging
from typing import List

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
        "For example, wav and flac are supported. All models from icefall "
        "uses 16 kHz training data. If the input sound file has a sample rate "
        "different from 16 kHz, it is resampled to 16 kHz. "
        "Only the first channel is used.",
    )

    return parser.parse_args()


async def run(server_addr: str, server_port: int, test_wavs: List[str]):
    async with websockets.connect(
        f"ws://{server_addr}:{server_port}"
    ) as websocket:  # noqa
        for test_wav in test_wavs:
            logging.info(f"Sending {test_wav}")
            wave, sample_rate = torchaudio.load(test_wav)

            if sample_rate != 16000:
                wave = torchaudio.functional.resample(
                    wave,
                    orig_freq=sample_rate,
                    new_freq=16000,
                )
                sample_rate = 16000

            wave = wave.squeeze(0).contiguous()

            # wave is a 1-D float32 tensor normalized to the range [-1, 1]
            # The format of the message sent to the server for each wave is
            #
            # - 4-byte in little endian specifying number of subsequent bytes
            #   to send
            # - one or more messages containing the data
            # - The last message is "Done"

            num_bytes = wave.numel() * wave.element_size()
            await websocket.send((num_bytes).to_bytes(4, "little", signed=True))

            frame_size = (2 ** 20) // 4  # max payload is 1MB
            sleep_time = 0.25
            start = 0
            while start < wave.numel():
                end = start + frame_size

                # reinterpret floats to bytes
                d = wave.numpy().data[start:end].tobytes()

                await websocket.send(d)
                await asyncio.sleep(sleep_time)  # in seconds

                start = end

            decoding_results = await websocket.recv()
            if decoding_results == "<EMPTY>":
                decoding_results = ""
            logging.info(f"{test_wav}\n{decoding_results}")
        await websocket.send("Done")


async def main():
    args = get_args()
    assert len(args.sound_files) > 0, "Empty sound files"

    server_addr = args.server_addr
    server_port = args.server_port

    max_retry_count = 5
    count = 0
    while count < max_retry_count:
        count += 1
        try:
            await run(server_addr, server_port, args.sound_files)
            break
        except websockets.exceptions.InvalidStatusCode as e:
            print(e.status_code)
            print(http.client.responses[e.status_code])
            print(e.headers)

            if e.status_code != http.HTTPStatus.SERVICE_UNAVAILABLE:
                raise
            await asyncio.sleep(2)
        except:  # noqa
            raise


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    logging.basicConfig(format=formatter, level=logging.INFO)
    asyncio.run(main())
