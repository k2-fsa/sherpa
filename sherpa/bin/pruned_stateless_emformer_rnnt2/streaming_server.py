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
"""

import argparse
import asyncio
import http
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
import websockets
from decode import Stream, stack_states, unstack_states

from sherpa import (RnntEmformerModel, fast_beam_search_one_best,
                    streaming_greedy_search)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="The server will listen on this port",
    )

    parser.add_argument(
        "--nn-model-filename",
        type=str,
        help="""The torchscript model. You can use
          icefall/egs/librispeech/ASR/pruned_transducer_statelessX/export.py \
                  --jit=1
        to generate this model.
        """,
    )

    parser.add_argument(
        "--bpe-model-filename",
        type=str,
        help="""The BPE model
        You can find it in the directory egs/librispeech/ASR/data/lang_bpe_xxx
        where xxx is the number of BPE tokens you used to train the model.
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

    parser.add_argument(
        "--beam",
        type=float,
        default=10.0,
        help="""A floating point value to calculate the cutoff score during beam
            search (i.e., `cutoff = max-score - beam`), which is the same as the
            `beam` in Kaldi.
            Used only when --decoding-method is fast_beam_search.
            """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is fast_beam_search.""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=32,
        help="""Used only when --decoding-method is fast_beam_search.""",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Decoding method to use. Currently, only greedy_search and
            fast_beam_search are implemented.
            """,
    )

    return parser.parse_args()


@torch.no_grad()
def run_model_and_do_search(
    server: "StreamingServer",
    stream_list: List[Stream],
) -> None:
    """Run the model on the given stream list and do greedy search.
    Args:
      server:
        An instance of `StreamingServer`.
      stream_list:
        A list of streams to be processed. It is changed in-place.
        That is, the attribute `states`, `decoder_out`, and `hyp` are
        updated in-place.
    """
    model = server.model
    device = model.device
    segment_length = server.segment_length
    chunk_length = server.chunk_length
    decoding_method = server.decoding_method

    batch_size = len(stream_list)

    state_list = []
    feature_list = []
    processed_frames_list = []
    if decoding_method == "greedy_search":
        decoder_out_list = []
        hyp_list = []
    else:
        rnnt_decoding_streams_list = []
        rnnt_decoding_config = server.rnnt_decoding_config

    for s in stream_list:
        if decoding_method == "greedy_search":
            decoder_out_list.append(s.decoder_out)
            hyp_list.append(s.hyp)
        elif decoding_method == "fast_beam_search":
            rnnt_decoding_streams_list.append(s.rnnt_decoding_stream)

        state_list.append(s.states)
        processed_frames_list.append(s.processed_frames)

        f = s.features[:chunk_length]
        s.features = s.features[segment_length:]
        b = torch.cat(f, dim=0)
        feature_list.append(b)

    features = torch.stack(feature_list, dim=0).to(device)
    states = stack_states(state_list)

    if decoding_method == "greedy_search":
        decoder_out = torch.cat(decoder_out_list, dim=0)

    features_length = torch.full(
        (batch_size,),
        fill_value=features.size(1),
        device=device,
        dtype=torch.int64,
    )

    processed_frames = torch.tensor(processed_frames_list, device=device)

    (encoder_out, encoder_out_lens, next_states) = model.encoder_streaming_forward(
        features=features,
        features_length=features_length,
        states=states,
    )
    if decoding_method == "fast_beam_search":
        processed_lens = processed_frames + encoder_out_lens
        next_hyp_list = fast_beam_search_one_best(
            model=model,
            encoder_out=encoder_out,
            processed_lens=processed_lens,
            rnnt_decoding_config=rnnt_decoding_config,
            rnnt_decoding_streams_list=rnnt_decoding_streams_list,
        )
    else:
        # Note: It does not return the next_encoder_out_len since
        # there are no paddings for streaming ASR. Each stream
        # has the same input number of frames, i.e., server.chunk_length.
        next_decoder_out, next_hyp_list = streaming_greedy_search(
            model=model,
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            hyps=hyp_list,
        )

    if decoding_method == "greedy_search":
        next_decoder_out_list = next_decoder_out.split(1)

    next_state_list = unstack_states(next_states)
    for i, s in enumerate(stream_list):
        s.states = next_state_list[i]
        s.processed_frames += encoder_out_lens[i]
        if decoding_method == "greedy_search":
            s.decoder_out = next_decoder_out_list[i]

        s.hyp = next_hyp_list[i]


class StreamingServer(object):
    def __init__(
        self,
        nn_model_filename: str,
        bpe_model_filename: str,
        beam: float,
        max_states: int,
        max_contexts: int,
        decoding_method: str,
        nn_pool_size: int,
        max_wait_ms: float,
        max_batch_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
    ):
        """
        Args:
          nn_model_filename:
            Path to the torchscript model
          bpe_model_filename:
            Path to the BPE model
          beam:
            The beam for fast_beam_search decoding.
          max_states:
            The max_states for fast_beam_search decoding.
          max_contexts:
            The max_contexts for fast_beam_search decoding.
          decoding_method:
            The decoding method to use, can be either greedy_search or fast_beam_search.
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
        """
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
        logging.info(f"Using device: {device}")

        self.model = RnntEmformerModel(nn_model_filename, device=device)

        # number of frames before subsampling
        self.segment_length = self.model.segment_length

        self.right_context_length = self.model.right_context_length

        # We add 3 here since the subsampling method is using
        # ((len - 1) // 2 - 1) // 2)
        self.chunk_length = self.segment_length + 3 + self.right_context_length

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_filename)

        self.context_size = self.model.context_size
        self.blank_id = self.model.blank_id
        self.vocab_size = self.model.vocab_size
        self.log_eps = math.log(1e-10)

        initial_states = self.model.get_encoder_init_states()
        self.initial_states = unstack_states(initial_states)[0]

        self.decoding_method = decoding_method
        assert self.decoding_method in ["greedy_search", "fast_beam_search"]

        self.initial_decoder_out = None
        self.decoding_graph = None
        if decoding_method == "fast_beam_search":
            self.rnnt_decoding_config = k2.RnntDecodingConfig(
                vocab_size=self.vocab_size,
                decoder_history_len=self.context_size,
                beam=beam,
                max_states=max_states,
                max_contexts=max_contexts,
            )
            self.decoding_graph = k2.trivial_graph(self.vocab_size - 1, device)
        else:
            decoder_input = torch.tensor(
                [[self.blank_id] * self.context_size],
                device=device,
                dtype=torch.int64,
            )
            initial_decoder_out = self.model.decoder_forward(decoder_input)
            self.initial_decoder_out = self.model.forward_decoder_proj(
                initial_decoder_out.squeeze(1)
            )

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

                    assert len(item[0].features) >= self.chunk_length, len(
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
                run_model_and_do_search,
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
            blank_id=self.blank_id,
            initial_states=self.initial_states,
            decoding_method=self.decoding_method,
            decoding_graph=self.decoding_graph,
            decoder_out=self.initial_decoder_out,
        )

        while True:
            samples = await self.recv_audio_samples(socket)
            if samples is None:
                break

            # TODO(fangjun): At present, we assume the sampling rate
            # of the received audio samples is always 16000.
            stream.accept_waveform(sampling_rate=16000, waveform=samples)

            while len(stream.features) > self.chunk_length:
                await self.compute_and_decode(stream)
                await socket.send(
                    f"{self.sp.decode(stream.hyp[self.context_size:] if self.decoding_method != 'fast_beam_search' else stream.hyp)}"
                )  # noqa

        stream.input_finished()
        while len(stream.features) > self.chunk_length:
            await self.compute_and_decode(stream)

        if len(stream.features) > 0:
            n = self.chunk_length - len(stream.features)
            stream.add_tail_paddings(n)
            await self.compute_and_decode(stream)
            stream.features = []

        result = self.sp.decode(stream.hyp[self.context_size :])  # noqa
        await socket.send(result)
        await socket.send("Done")

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
    args = get_args()

    logging.info(vars(args))

    port = args.port
    nn_model_filename = args.nn_model_filename
    bpe_model_filename = args.bpe_model_filename
    beam = args.beam
    max_states = args.max_states
    max_contexts = args.max_contexts
    decoding_method = args.decoding_method
    nn_pool_size = args.nn_pool_size
    max_batch_size = args.max_batch_size
    max_wait_ms = args.max_wait_ms
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections

    server = StreamingServer(
        nn_model_filename=nn_model_filename,
        bpe_model_filename=bpe_model_filename,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        decoding_method=decoding_method,
        nn_pool_size=nn_pool_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
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
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
