#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#                2023  Nvidia              (authors: Yuekai Zhang)
#                2023  Recurrent.ai        (authors: Songtao Shi)
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
This script supports to load manifest files in kaldi format and sends it to the server
for decoding, in parallel.

Usage:
# For offline icefall server
python3 client.py \
    --compute-cer  # For Chinese, we use CER to evaluate the model 

# For streaming icefall server
python3 client.py \
    --streaming \
    --compute-cer

# For simulate streaming mode icefall server
python3 client.py \
    --simulate-streaming \
    --compute-cer

# For offline wenet server
python3 client.py \
    --server-addr localhost \
    --compute-cer \
    --model-name attention_rescoring \
    --num-tasks 300 \
    --manifest-dir ./datasets/aishell1_test

# For streaming wenet server
python3 client.py \
    --server-addr localhost \
    --streaming \
    --compute-cer \
    --context 7 \
    --model-name streaming_wenet \
    --num-tasks 300 \
    --manifest-dir ./datasets/aishell1_test

# For simulate streaming mode wenet server
python3 client.py \
    --server-addr localhost \
    --simulate-streaming \
    --compute-cer \
    --context 7 \
    --model-name streaming_wenet \
    --num-tasks 300 \
    --manifest-dir ./datasets/aishell1_test

# For offlien paraformer server
python3 client.py \
    --server-addr localhost \
    --compute-cer \
    --model-name infer_pipeline \
    --num-tasks $num_task \
    --manifest-dir ./datasets/aishell1_test

# For offlien whisper server
python3 client.py \
    --server-addr localhost \
    --model-name whisper \
    --num-tasks $num_task \
    --text-prompt "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --manifest-dir ./datasets/mini_en

# For offline sensevoice server
python3 client.py \
    --server-addr localhost \
    --server-port 10086 \
    --model-name sensevoice \
    --num-tasks $num_task \
    --batch-size $bach_size \
    --manifest-dir ./datasets/mini_zh

# For offline whisper_qwen2 server
python3 client.py \
    --server-addr localhost \
    --model-name infer_bls \
    --num-tasks $num_task \
    --manifest-dir ./datasets/mini_zh \
    --compute-cer

# huggingface dataset
dataset_name=yuekai/aishell
subset_name=test
split_name=test
num_task=32
python3 client.py \
    --server-addr localhost \
    --model-name infer_bls \
    --num-tasks $num_task \
    --text-prompt "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --log-dir ./log_sherpa_multi_hans_whisper_large_ifb_$num_task \
    --compute-cer
"""

import argparse
import asyncio
import json
import math
import os
import re
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype

from utils import (
    download_and_extract,
    store_transcripts,
    write_error_stats,
    write_triton_stats,
)

DEFAULT_MANIFEST_DIR = "./datasets/aishell1_test"


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
        help="Grpc port of the triton server, default is 8001",
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=DEFAULT_MANIFEST_DIR,
        help="Path to the manifest dir which includes wav.scp trans.txt files.",
    )

    parser.add_argument(
        "--audio-path",
        type=str,
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--text-prompt",
        type=str,
        default="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        help="e.g. <|startofprev|>My hot words<|startoftranscript|><|en|><|transcribe|><|notimestamps|>, please check https://arxiv.org/pdf/2305.11095.pdf also.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="preprocess_flow_matching",
        choices=[
            "transducer",
            "attention_rescoring",
            "streaming_wenet",
            "infer_pipeline",
            "whisper",
            "whisper_bls",
            "sensevoice",
            "infer_bls",
            "preprocess_flow_matching",
        ],
        help="triton model_repo module name to request: transducer for k2, attention_rescoring for wenet offline, streaming_wenet for wenet streaming, infer_pipeline for paraformer large offline",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of concurrent tasks for sending",
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

    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="""True for streaming ASR.
        """,
    )

    parser.add_argument(
        "--simulate-streaming",
        action="store_true",
        default=False,
        help="""True for strictly simulate streaming ASR.
        Threads will sleep to simulate the real speaking scene.
        """,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        default=16,
        help="Parameter for streaming ASR, chunk size default is 16",
    )

    parser.add_argument(
        "--context",
        type=int,
        required=False,
        default=-1,
        help="subsampling context for wenet",
    )

    parser.add_argument(
        "--encoder_right_context",
        type=int,
        required=False,
        default=2,
        help="encoder right context for k2 streaming",
    )

    parser.add_argument(
        "--subsampling",
        type=int,
        required=False,
        default=4,
        help="subsampling rate",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        default="./tmp",
        help="log directory",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch_size per request for offline mode.",
    )

    parser.add_argument("--huggingface_dataset", type=str, default=None)
    parser.add_argument(
        "--subset_name",
        type=str,
        default=None,
        help="dataset configuration name in the dataset, see https://huggingface.co/docs/datasets/v3.0.0/en/package_reference/loading_methods#datasets.load_dataset",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="dataset split name, default is 'test'",
    )
    return parser.parse_args()


def load_audio(wav_path):
    waveform, sample_rate = sf.read(wav_path)
    return waveform, sample_rate
    if sample_rate == 16000:
        return waveform, sample_rate
    elif sample_rate == 8000:
        from scipy.signal import resample

        # Upsample from 8k to 16k
        num_samples = int(len(waveform) * (16000 / 8000))
        upsampled_waveform = resample(waveform, num_samples)
        return upsampled_waveform, 16000
    elif sample_rate == 48000:
        from scipy.signal import resample

        # Downsample from 48k to 16k
        num_samples = int(len(waveform) * (16000 / 48000))
        downsampled_waveform = resample(waveform, num_samples)
        return downsampled_waveform, 16000
    else:
        raise ValueError(f"Only support 8k, 16k and 48k sample rates, but got {sample_rate}")


async def send_whisper(
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    compute_cer: bool,
    model_name: str,
    padding_duration: int = 30,
    whisper_prompt: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
):
    total_duration = 0.0
    results = []
    latency_data = []
    task_id = int(name[5:])

    audio_filepath='./assets/wgs-f5tts_mono.wav'
    dps = [{"audio_filepath": audio_filepath}]
    for i, dp in enumerate(dps):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(dps)}")

        waveform, sample_rate = load_audio(dp["audio_filepath"])
        duration = len(waveform) / sample_rate

        reference_text = "那到时候再给你打电话，麻烦你注意接听。"
        target_text = "这点请您放心，估计是我的号码被标记了，请问您是沈沈吗？"

        # padding to nearset 10 seconds
        # samples = np.zeros(
        #     (
        #         1,
        #         padding_duration
        #         * sample_rate
        #         * ((int(duration) // padding_duration) + 1),
        #     ),
        #     dtype=np.float32,
        # )

        # samples[0, : len(waveform)] = waveform
        # expand_dims
        samples = waveform.reshape(1, -1).astype(np.float32)

        lengths = np.array([[len(waveform)]], dtype=np.int32)
        # print(f"lengths: {lengths}")

        inputs = [
            protocol_client.InferInput(
                "reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)
            ),
            protocol_client.InferInput(
                "reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)
            ),
            protocol_client.InferInput("reference_text", [1, 1], "BYTES"),
            protocol_client.InferInput("target_text", [1, 1], "BYTES")
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)

        input_data_numpy = np.array([reference_text], dtype=object)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[2].set_data_from_numpy(input_data_numpy)

        input_data_numpy = np.array([target_text], dtype=object)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[3].set_data_from_numpy(input_data_numpy)

        outputs = [protocol_client.InferRequestedOutput("target_mel")]
        sequence_id = 100000000 + i + task_id * 10
        start = time.time()
        response = await triton_client.infer(
            model_name, inputs, request_id=str(sequence_id), outputs=outputs
        )

        # decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        # if type(decoding_results) == np.ndarray:
        #     decoding_results = b" ".join(decoding_results).decode("utf-8")
        # else:
        #     # For wenet
        #     decoding_results = decoding_results.decode("utf-8")
        # end = time.time() - start
        # latency_data.append((end, duration))
        # total_duration += duration

    return total_duration, results, latency_data


async def main():
    args = get_args()
    url = f"{args.server_addr}:{args.server_port}"

    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    protocol_client = grpcclient

   
    tasks = []
    start_time = time.time()
    for i in range(args.num_tasks):
        task = asyncio.create_task(
            send_whisper(
                name=f"task-{i}",
                triton_client=triton_client,
                protocol_client=protocol_client,
                log_interval=args.log_interval,
                compute_cer=args.compute_cer,
                model_name=args.model_name,
                whisper_prompt=args.text_prompt,
            )
        )
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    # results = []
    # total_duration = 0.0
    # latency_data = []
    # for ans in ans_list:
    #     total_duration += ans[0]
    #     results += ans[1]
    #     latency_data += ans[2]

    # rtf = elapsed / total_duration

    # s = f"RTF: {rtf:.4f}\n"
    # s += f"total_duration: {total_duration:.3f} seconds\n"
    # s += f"({total_duration/3600:.2f} hours)\n"
    # s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"

    # latency_list = [chunk_end for (chunk_end, chunk_duration) in latency_data]
    # latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    # latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
    # s += f"latency_variance: {latency_variance:.2f}\n"
    # s += f"latency_50_percentile_ms: {np.percentile(latency_list, 50) * 1000.0:.2f}\n"
    # s += f"latency_90_percentile_ms: {np.percentile(latency_list, 90) * 1000.0:.2f}\n"
    # s += f"latency_95_percentile_ms: {np.percentile(latency_list, 95) * 1000.0:.2f}\n"
    # s += f"latency_99_percentile_ms: {np.percentile(latency_list, 99) * 1000.0:.2f}\n"
    # s += f"average_latency_ms: {latency_ms:.2f}\n"

    # print(s)
    # os.makedirs(args.log_dir, exist_ok=True)
    # name = Path(args.manifest_dir).stem.split(".")[0]
    # with open(f"{args.log_dir}/rtf-{name}.txt", "w") as f:
    #     f.write(s)
    # results = sorted(results)
    # store_transcripts(filename=f"{args.log_dir}/recogs-{name}.txt", texts=results)

    # with open(f"{args.log_dir}/errs-{name}.txt", "w") as f:
    #     write_error_stats(f, "test-set", results, enable_log=True)

    # with open(f"{args.log_dir}/errs-{name}.txt", "r") as f:
    #     print(f.readline())  # WER
    #     print(f.readline())  # Detailed errors

    #stats = await triton_client.get_inference_statistics(model_name="", as_json=True)
    #write_triton_stats(stats, f"{args.log_dir}/stats_summary-{name}.txt")


if __name__ == "__main__":
    asyncio.run(main())
