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
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import sentencepiece as spm
import torch
import websockets
from sherpa import RnntEmformerModel, streaming_greedy_search

from decode import Stream, stack_states, unstack_states

DEFAULT_NN_MODEL_FILENAME = "/ceph-fj/fangjun/open-source-2/icefall-streaming-2/egs/librispeech/ASR/pruned_stateless_emformer_rnnt2/exp-full/cpu_jit-epoch-39-avg-6-use-averaged-model-1.pt"  # noqa
DEFAULT_BPE_MODEL_FILENAME = "/ceph-fj/fangjun/open-source-2/icefall-streaming-2/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"  # noqa

TEST_WAV = "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav"
TEST_WAV = "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0001.wav"
#  TEST_WAV = "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0002.wav"


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
        default=DEFAULT_NN_MODEL_FILENAME,
        help="""The torchscript model. You can use
          icefall/egs/librispeech/ASR/pruned_transducer_statelessX/export.py --jit=1
        to generate this model.
        """,
    )

    parser.add_argument(
        "--bpe-model-filename",
        type=str,
        default=DEFAULT_BPE_MODEL_FILENAME,
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

    return parser.parse_args()


def run_model_and_do_greedy_search(
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

    batch_size = len(stream_list)

    state_list = []
    decoder_out_list = []
    hyp_list = []
    feature_list = []
    for s in stream_list:
        state_list.append(s.states)
        decoder_out_list.append(s.decoder_out)
        hyp_list.append(s.hyp)

        f = s.features[:chunk_length]
        s.features = s.features[segment_length:]

        b = torch.cat(f, dim=0)
        feature_list.append(b)

    features = torch.stack(feature_list, dim=0).to(device)
    states = stack_states(state_list)
    decoder_out = torch.cat(decoder_out_list, dim=0)

    features_length = torch.full(
        (batch_size,),
        fill_value=features.size(1),
        device=device,
    )

    (encoder_out, next_states) = model.encoder_streaming_forward(
        features,
        features_length,
        states,
    )

    # Note: It does not return the next_encoder_out_len since
    # there are no paddings for streaming ASR. Each stream
    # has the same input number of frames, i.e., server.chunk_length.
    next_decoder_out, next_hyp_list = streaming_greedy_search(
        model=model,
        encoder_out=encoder_out,
        decoder_out=decoder_out,
        hyps=hyp_list,
    )

    next_state_list = unstack_states(next_states)
    next_decoder_out_list = next_decoder_out.split(1)
    for i, s in enumerate(stream_list):
        s.states = next_state_list[i]
        s.decoder_out = next_decoder_out_list[i]
        s.hyp = next_hyp_list[i]


class StreamingServer(object):
    def __init__(
        self,
        nn_model_filename: str,
        bpe_model_filename: str,
        nn_pool_size: int,
        max_wait_ms: float,
        max_batch_size: int,
    ):
        """
        Args:
          nn_model_filename:
            Path to the torchscript model
          bpe_model_filename:
            Path to the BPE model
          nn_pool_size:
            Number of threads for the thread pool that is responsible for
            neural network computation and decoding.
          max_wait_ms:
            Max wait time in milliseconds in order to build a batch of
            `batch_size`.
          max_batch_size:
            Max batch size for inference.
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
        self.chunk_length = (self.segment_length + 3) + self.right_context_length

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_filename)

        self.context_size = self.model.context_size
        self.blank_id = self.model.blank_id
        self.log_eps = math.log(1e-10)

        initial_states = self.model.get_encoder_init_states()
        self.initial_states = unstack_states(initial_states)[0]
        decoder_input = torch.tensor(
            [[self.blank_id] * self.context_size],
            device=device,
            dtype=torch.int64,
        )
        self.initial_decoder_out = self.model.decoder_forward(decoder_input).squeeze(1)

        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )

        self.stream_queue = asyncio.Queue()
        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size

    async def stream_consumer_task(self):
        """The function extract streams from the queue, batches them up, sends
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
                run_model_and_do_greedy_search,
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

    async def run(self, port: int):
        task = asyncio.create_task(self.stream_consumer_task())

        async with websockets.serve(self.handle_connection, "", port):
            await asyncio.Future()  # run forever

    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and sends
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        logging.info(f"Connected: {socket.remote_address}")
        stream = Stream(
            context_size=self.context_size,
            blank_id=self.blank_id,
            initial_states=self.initial_states,
            decoder_out=self.initial_decoder_out,
        )

        last = b""
        while True:
            samples, last = await self.recv_audio_samples(socket, last)
            if samples is None:
                break

            # TODO(fangjun): At present, we assume the sampling rate
            # of the received audio samples is always 16000.
            stream.accept_waveform(sampling_rate=16000, waveform=samples)

            while len(stream.features) > self.chunk_length:
                await self.compute_and_decode(stream)
                await socket.send(f"{self.sp.decode(stream.hyp[self.context_size:])}")

        stream.input_finished()
        while len(stream.features) > self.chunk_length:
            await self.compute_and_decode(stream)

        if len(stream.features) > 0:
            n = self.chunk_length - len(stream.features)
            stream.add_tail_paddings(n)
            await self.compute_and_decode(stream)
            stream.features = []

        result = self.sp.decode(stream.hyp[self.context_size :])
        await socket.send(result)
        await socket.send("Done")

        logging.info(f"Disconnected: {socket.remote_address}")

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
        last: Optional[bytes] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[bytes]]:
        """Receives a tensor from the client.

        The message from the client contains two parts: header and payload

            - the header contains 8 bytes in little endian format, specifying
              the number of bytes in the payload.

            - the payload contains either a binary representation of the 1-D
              torch.float32 tensor or the bytes object b"Done" which means
              the end of utterance.

        Args:
          socket:
            The socket for communicating with the client.
          last:
            Previous received content.
        Returns:
          Return a tuple containing:
            - A 1-D torch.float32 tensor containing the audio samples
            - Data for the next chunk, if any
         or return a tuple (None, None) meaning the end of utterance.
        """
        header_len = 8

        if last is None:
            last = b""

        async def receive_header():
            buf = last
            async for message in socket:
                buf += message
                if len(buf) >= header_len:
                    break
            if buf:
                header = buf[:header_len]
                remaining = buf[header_len:]
            else:
                header = None
                remaining = None

            return header, remaining

        header, received = await receive_header()

        if header is None:
            return None, None

        expected_num_bytes = int.from_bytes(header, "little", signed=True)

        async for message in socket:
            received += message
            if len(received) >= expected_num_bytes:
                break

        if not received or received == b"Done":
            return None, None

        this_chunk = received[:expected_num_bytes]
        next_chunk = received[expected_num_bytes:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # PyTorch warns that the underlying buffer is not writable.
            # We ignore it here as we are not going to write it anyway.
            return torch.frombuffer(this_chunk, dtype=torch.float32), next_chunk


@torch.no_grad()
def main():
    args = get_args()

    logging.info(vars(args))

    port = args.port
    nn_model_filename = args.nn_model_filename
    bpe_model_filename = args.bpe_model_filename
    nn_pool_size = args.nn_pool_size
    max_batch_size = args.max_batch_size
    max_wait_ms = args.max_wait_ms

    server = StreamingServer(
        nn_model_filename=nn_model_filename,
        bpe_model_filename=bpe_model_filename,
        nn_pool_size=nn_pool_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
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
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
