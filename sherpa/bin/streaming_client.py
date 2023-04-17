#!/usr/bin/env python3
# Copyright      2022-2023  Xiaomi Corp.

"""
A client for streaming ASR recognition.

Usage:
    ./streaming_client.py \
      --server-addr localhost \
      --server-port 6006 \
      /path/to/foo.wav \
      /path/to/bar.wav

(Note: You have to first start the server before starting the client)
See ./streaming_server.py
for how to start the server
"""
import argparse
import asyncio
import http
import json
import logging

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


async def receive_results(socket: websockets.WebSocketServerProtocol):
    global done
    ans = []
    async for message in socket:
        result = json.loads(message)

        method = result["method"]
        segment = result["segment"]
        is_final = result["final"]
        text = result["text"]
        tokens = result["tokens"]
        timestamps = result["timestamps"]

        if is_final:
            ans.append(
                dict(
                    method=method,
                    segment=segment,
                    text=text,
                    tokens=tokens,
                    timestamps=timestamps,
                )
            )
            logging.info(f"Final result of segment {segment}: {text}")
            continue

        last_10_words = text.split()[-10:]
        last_10_words = " ".join(last_10_words)
        logging.info(
            f"Partial result of segment {segment} (last 10 words): "
            f"{last_10_words}"
        )

    return ans


async def run(server_addr: str, server_port: int, test_wav: str):
    async with websockets.connect(
        f"ws://{server_addr}:{server_port}"
    ) as websocket:  # noqa
        logging.info(f"Sending {test_wav}")
        wave, sample_rate = torchaudio.load(test_wav)
        # You have to ensure that sample_rate equals to
        # the argument --audio-sample-rate that you used to
        # start streaming_server.py
        logging.info(f"sample_rate: {sample_rate}")

        wave = wave.squeeze(0)
        receive_task = asyncio.create_task(receive_results(websocket))

        frame_size = 4096
        sleep_time = frame_size / sample_rate  # in seconds
        start = 0
        while start < wave.numel():
            end = start + min(frame_size, wave.numel() - start)
            d = wave.numpy().data[start:end]

            await websocket.send(d)
            await asyncio.sleep(sleep_time)  # in seconds

            start += frame_size

        await websocket.send("Done")
        decoding_results = await receive_task
        s = ""
        for r in decoding_results:
            s += f"method: {r['method']}\n"
            s += f"segment: {r['segment']}\n"
            s += f"text: {r['text']}\n"

            token_time = []
            for token, time in zip(r["tokens"], r["timestamps"]):
                token_time.append((token, time))

            s += f"timestamps: {r['timestamps']}\n"
            s += f"(token, time): {token_time}\n"
        logging.info(f"{test_wav}\n{s}")


async def main():
    args = get_args()
    assert len(args.sound_files) > 0, "Empty sound files"

    server_addr = args.server_addr
    server_port = args.server_port

    max_retry_count = 5
    for sound_file in args.sound_files:
        count = 0
        while count < max_retry_count:
            count += 1
            try:
                await run(server_addr, server_port, sound_file)
                break
            except websockets.exceptions.InvalidStatusCode as e:
                print(e.status_code)
                print(http.client.responses[e.status_code])
                print(e.headers)

                if e.status_code != http.HTTPStatus.SERVICE_UNAVAILABLE:
                    raise
                await asyncio.sleep(2)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    logging.basicConfig(format=formatter, level=logging.INFO)
    asyncio.run(main())
