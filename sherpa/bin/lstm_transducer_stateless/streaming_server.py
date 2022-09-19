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
A server for streaming ASR recognition. By streaming it means the audio samples
are coming in real-time. You don't need to wait until all audio samples are
captured before sending them for recognition.

It supports multiple clients sending at the same time.

Usage:
    ./streaming_server.py --help

    ./streaming_server.py

Please refer to
https://k2-fsa.github.io/sherpa/python/streaming_asr/lstm/index.html
for details.
"""

import argparse
import asyncio
import http
import json
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
import websockets
from beam_search import FastBeamSearch, GreedySearch, ModifiedBeamSearch
from stream import Stream

from sherpa import (
    OnlineEndpointConfig,
    RnntLstmModel,
    add_beam_search_arguments,
    add_online_endpoint_arguments,
)


def get_args():
    beam_search_parser = add_beam_search_arguments()
    online_endpoint_parser = add_online_endpoint_arguments()
    parser = argparse.ArgumentParser(
        parents=[beam_search_parser, online_endpoint_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="The server will listen on this port",
    )

    parser.add_argument(
        "--nn-encoder-filename",
        type=str,
        required=True,
        help="""The torchscript encoder. You can use
          icefall/egs/librispeech/ASR/lstm_transducer_stateless/export.py \
                  --jit-trace=1
        to generate this model.
        """,
    )

    parser.add_argument(
        "--nn-decoder-filename",
        type=str,
        required=True,
        help="""The torchscript decoder. You can use
          icefall/egs/librispeech/ASR/lstm_transducer_stateless/export.py \
                  --jit-trace=1
        to generate this model.
        """,
    )

    parser.add_argument(
        "--nn-joiner-filename",
        type=str,
        required=True,
        help="""The torchscript decoder. You can use
          icefall/egs/librispeech/ASR/lstm_transducer_stateless/export.py \
                  --jit-trace=1
        to generate this model.
        """,
    )

    parser.add_argument(
        "--bpe-model-filename",
        type=str,
        help="""The BPE model
        You can find it in the directory egs/librispeech/ASR/data/lang_bpe_xxx
        where xxx is the number of BPE tokens you used to train the model.
        Note: You don't need to provide it if you provide `--token-filename`.
        """,
    )

    parser.add_argument(
        "--token-filename",
        type=str,
        help="""Filename for tokens.txt
        For instance, you can find it in the directory
        egs/aishell/ASR/data/lang_char/tokens.txt
        or
        egs/wenetspeech/ASR/data/lang_char/tokens.txt
        from icefall
        Note: You don't need to provide it if you provide `--bpe-model`
        """,
    )

    parser.add_argument(
        "--nn-pool-size",
        type=int,
        default=1,
        help="Number of threads for NN computation and decoding.",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=50,
        help="""Max batch size for computation. Note if there are not enough
        requests in the queue, it will wait for max_wait_ms time. After that,
        even if there are not enough requests, it still sends the
        available requests in the queue for computation.
        """,
    )

    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=10,
        help="""Max time in millisecond to wait to build batches for inference.
        If there are not enough requests in the stream queue to build a batch
        of max_batch_size, it waits up to this time before fetching available
        requests for computation.
        """,
    )

    parser.add_argument(
        "--max-message-size",
        type=int,
        default=(1 << 20),
        help="""Max message size in bytes.
        The max size per message cannot exceed this limit.
        """,
    )

    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=32,
        help="Max number of messages in the queue for each connection.",
    )

    parser.add_argument(
        "--max-active-connections",
        type=int,
        default=500,
        help="""Maximum number of active connections. The server will refuse
        to accept new connections once the current number of active connections
        equals to this limit.
        """,
    )

    return (
        parser.parse_args(),
        beam_search_parser.parse_known_args()[0],
        online_endpoint_parser.parse_known_args()[0],
    )


class StreamingServer(object):
    def __init__(
        self,
        nn_encoder_filename: str,
        nn_decoder_filename: str,
        nn_joiner_filename: str,
        bpe_model_filename: Optional[str],
        token_filename: Optional[str],
        nn_pool_size: int,
        max_wait_ms: float,
        max_batch_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
        beam_search_params: dict,
        online_endpoint_config: OnlineEndpointConfig,
    ):
        """
        Args:
          nn_encoder_filename:
            Path to the torchscript encoder.
          nn_decoder_filename:
            Path to the torchscript decoder.
          nn_joiner_filename:
            Path to the torchscript joiner.
          bpe_model_filename:
            Path to the BPE model. If it is None, you have to provide
            `token_filename`.
          token_filename:
            Path to tokens.txt. If it is None, you have to provide
            `bpe_model_filename`.
          nn_pool_size:
            Number of threads for the thread pool that is responsible for
            neural network computation and decoding.
          max_wait_ms:
            Max wait time in milliseconds in order to build a batch of
            `batch_size`.
          max_batch_size:
            Max batch size for inference.
          max_message_size:
            Max size in bytes per message.
          max_queue_size:
            Max number of messages in the queue for each connection.
          max_active_connections:
            Max number of active connections. Once number of active client
            equals to this limit, the server refuses to accept new connections.
          beam_search_params:
            Dictionary containing all the parameters for beam search.
          online_endpoint_config:
            Config for endpointing.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
        logging.info(f"Using device: {device}")

        self.model = RnntLstmModel(
            nn_encoder_filename,
            nn_decoder_filename,
            nn_joiner_filename,
            device=device,
        )

        self.subsampling_factor = self.model.subsampling_factor
        # number of frames before subsampling
        self.chunk_length = self.model.subsampling_factor

        # We add 5 here since the subsampling method is using
        # ((len - 3) // 2 - 1) // 2)
        self.chunk_length_pad = self.chunk_length + self.model.pad_length

        if bpe_model_filename:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_model_filename)
        else:
            self.token_table = k2.SymbolTable.from_file(token_filename)

        self.context_size = self.model.context_size
        self.blank_id = self.model.blank_id
        self.vocab_size = self.model.vocab_size
        self.log_eps = math.log(1e-10)

        self.initial_states = self.model.get_encoder_init_states()

        # Add these params after loading the Conv-Emformer model
        beam_search_params["vocab_size"] = self.vocab_size
        beam_search_params["context_size"] = self.context_size
        beam_search_params["blank_id"] = self.blank_id

        decoding_method = beam_search_params["decoding_method"]
        if decoding_method.startswith("fast_beam_search"):
            self.beam_search = FastBeamSearch(
                beam_search_params=beam_search_params,
                device=device,
            )
        elif decoding_method == "greedy_search":
            self.beam_search = GreedySearch(
                self.model,
                beam_search_params,
                device,
            )
        elif decoding_method == "modified_beam_search":
            self.beam_search = ModifiedBeamSearch(beam_search_params)
        else:
            raise ValueError(
                f"Decoding method {decoding_method} is not supported."
            )

        if hasattr(self, "sp"):
            self.beam_search.sp = self.sp
        else:
            self.beam_search.token_table = self.token_table

        self.online_endpoint_config = online_endpoint_config

        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )

        self.stream_queue = asyncio.Queue()
        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        self.max_active_connections = max_active_connections

        self.current_active_connections = 0

    async def warmup(self) -> None:
        """Do warmup to the torchscript model to decrease the waiting time
        of the first request.

        See https://github.com/k2-fsa/sherpa/pull/100 for details
        """
        logging.info("Warmup start")
        stream = Stream(
            context_size=self.context_size,
            subsampling_factor=self.subsampling_factor,
            initial_states=self.initial_states,
        )
        self.beam_search.init_stream(stream)

        samples = torch.rand(16000 * 1, dtype=torch.float32)  # 1 second
        stream.accept_waveform(sampling_rate=16000, waveform=samples)

        while len(stream.features) > self.chunk_length_pad:
            await self.compute_and_decode(stream)

        logging.info("Warmup done")

    async def stream_consumer_task(self):
        """This function extracts streams from the queue, batches them up, sends
        them to the RNN-T model for computation and decoding.
        """
        while True:
            if self.stream_queue.empty():
                await asyncio.sleep(self.max_wait_ms / 1000)
                continue

            batch = []
            try:
                while len(batch) < self.max_batch_size:
                    item = self.stream_queue.get_nowait()

                    assert len(item[0].features) >= self.chunk_length_pad, len(
                        item[0].features
                    )

                    batch.append(item)
            except asyncio.QueueEmpty:
                pass
            stream_list = [b[0] for b in batch]
            future_list = [b[1] for b in batch]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.nn_pool,
                self.beam_search.process,
                self,
                stream_list,
            )

            for f in future_list:
                self.stream_queue.task_done()
                f.set_result(None)

    async def compute_and_decode(
        self,
        stream: Stream,
    ) -> None:
        """Put the stream into the queue and wait it to be processed by the
        consumer task.

        Args:
          stream:
            The stream to be processed. Note: It is changed in-place.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.stream_queue.put((stream, future))
        await future

    async def process_request(
        self,
        unused_path: str,
        unused_request_headers: websockets.Headers,
    ) -> Optional[Tuple[http.HTTPStatus, websockets.Headers, bytes]]:
        if self.current_active_connections < self.max_active_connections:
            self.current_active_connections += 1
            return None

        # Refuse new connections
        status = http.HTTPStatus.SERVICE_UNAVAILABLE  # 503
        header = {"Hint": "The server is overloaded. Please retry later."}
        response = b"The server is busy. Please retry later."

        return status, header, response

    async def run(self, port: int):
        task = asyncio.create_task(self.stream_consumer_task())
        await self.warmup()

        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=self.max_message_size,
            max_queue=self.max_queue_size,
            process_request=self.process_request,
        ):
            await asyncio.Future()  # run forever

        await task  # not reachable

    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        try:
            await self.handle_connection_impl(socket)
        finally:
            # Decrement so that it can accept new connections
            self.current_active_connections -= 1

            logging.info(
                f"Disconnected: {socket.remote_address}. "
                f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
            )

    async def handle_connection_impl(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        logging.info(
            f"Connected: {socket.remote_address}. "
            f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
        )

        stream = Stream(
            context_size=self.context_size,
            initial_states=self.initial_states,
            subsampling_factor=self.subsampling_factor,
        )

        self.beam_search.init_stream(stream)

        while True:
            samples = await self.recv_audio_samples(socket)
            if samples is None:
                break

            # TODO(fangjun): At present, we assume the sampling rate
            # of the received audio samples is always 16000.
            stream.accept_waveform(sampling_rate=16000, waveform=samples)

            while len(stream.features) > self.chunk_length_pad:
                await self.compute_and_decode(stream)
                hyp = self.beam_search.get_texts(stream)

                segment = stream.segment
                is_final = stream.endpoint_detected(self.online_endpoint_config)
                if is_final:
                    self.beam_search.init_stream(stream)

                message = {
                    "segment": segment,
                    "text": hyp,
                    "final": is_final,
                }

                await socket.send(json.dumps(message))

        stream.input_finished()
        while len(stream.features) > self.chunk_length_pad:
            await self.compute_and_decode(stream)

        if len(stream.features) > 0:
            n = self.chunk_length_pad - len(stream.features)
            stream.add_tail_paddings(n)
            await self.compute_and_decode(stream)
            stream.features = []

        hyp = self.beam_search.get_texts(stream)

        message = {
            "segment": stream.segment,
            "text": hyp,
            "final": True,  # end of connection, always set final to True
        }

        await socket.send(json.dumps(message))

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Optional[torch.Tensor]:
        """Receives a tensor from the client.

        Each message contains either a bytes buffer containing audio samples
        in 16 kHz or contains b"Done" meaning the end of utterance.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a 1-D torch.float32 tensor containing the audio samples or
          return None.
        """
        message = await socket.recv()
        if message == b"Done":
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # PyTorch warns that the underlying buffer is not writable.
            # We ignore it here as we are not going to write it anyway.
            if hasattr(torch, "frombuffer"):
                # Note: torch.frombuffer is available only in torch>= 1.10
                return torch.frombuffer(message, dtype=torch.float32)
            else:
                array = np.frombuffer(message, dtype=np.float32)
                return torch.from_numpy(array)


@torch.no_grad()
def main():
    args, beam_search_parser, online_endpoint_parser = get_args()

    beam_search_params = vars(beam_search_parser)
    logging.info(beam_search_params)

    online_endpoint_params = vars(online_endpoint_parser)
    logging.info(online_endpoint_params)

    online_endpoint_config = OnlineEndpointConfig.from_args(
        online_endpoint_params
    )

    logging.info(vars(args))

    port = args.port
    nn_encoder_filename = args.nn_encoder_filename
    nn_decoder_filename = args.nn_decoder_filename
    nn_joiner_filename = args.nn_joiner_filename
    bpe_model_filename = args.bpe_model_filename
    token_filename = args.token_filename
    nn_pool_size = args.nn_pool_size
    max_batch_size = args.max_batch_size
    max_wait_ms = args.max_wait_ms
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections

    if beam_search_params["decoding_method"] == "modified_beam_search":
        assert beam_search_params["num_active_paths"] >= 1, beam_search_params[
            "num_active_paths"
        ]

    if bpe_model_filename:
        assert token_filename is None, (
            "You need to provide either --bpe-model-filename or "
            "--token-filename parameter. But not both."
        )

    if token_filename:
        assert bpe_model_filename is None, (
            "You need to provide either --bpe-model-filename or "
            "--token-filename parameter. But not both."
        )

    assert bpe_model_filename or token_filename, (
        "You need to provide either --bpe-model-filename or "
        "--token-filename parameter. But not both."
    )

    server = StreamingServer(
        nn_encoder_filename=nn_encoder_filename,
        nn_decoder_filename=nn_decoder_filename,
        nn_joiner_filename=nn_joiner_filename,
        bpe_model_filename=bpe_model_filename,
        token_filename=token_filename,
        nn_pool_size=nn_pool_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
        beam_search_params=beam_search_params,
        online_endpoint_config=online_endpoint_config,
    )
    asyncio.run(server.run(port))


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# See https://github.com/pytorch/pytorch/issues/38342
# and https://github.com/pytorch/pytorch/issues/33354
#
# If we don't do this, the delay increases whenever there is
# a new request that changes the actual batch size.
# If you use `py-spy dump --pid <server-pid> --native`, you will
# see a lot of time is spent in re-compiling the torch script model.
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
"""
// Use the following in C++
torch::jit::getExecutorMode() = false;
torch::jit::getProfilingMode() = false;
torch::jit::setGraphExecutorOptimize(false);
"""

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
