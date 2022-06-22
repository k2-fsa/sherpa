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
A server for offline ASR recognition. Offline means you send all the content
of the audio for recognition. It supports multiple clients sending at
the same time.

Usage:
    ./offline_server.py --help

    ./offline_server.py
"""

import argparse
import asyncio
import functools
import http
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import k2
import kaldifeat
import numpy as np
import sentencepiece as spm
import torch
import websockets
from _sherpa import RnntConformerModel, greedy_search, modified_beam_search
from torch.nn.utils.rnn import pad_sequence

LOG_EPS = math.log(1e-10)


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
        "--num-device",
        type=int,
        default=1,
        help="""Number of GPU devices to use. Set it to 0 to use CPU
        for computation. If positive, then GPUs with ID 0, 1, ..., num_device-1
        will be used for computation. You can use the environment variable
        CUDA_VISIBLE_DEVICES to map available GPU devices.
        """,
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=25,
        help="""Max batch size for computation. Note if there are not enough
        requests in the queue, it will wait for max_wait_ms time. After that,
        even if there are not enough requests, it still sends the
        available requests in the queue for computation.
        """,
    )

    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=5,
        help="""Max time in millisecond to wait to build batches for inference.
        If there are not enough requests in the feature queue to build a batch
        of max_batch_size, it waits up to this time before fetching available
        requests for computation.
        """,
    )

    parser.add_argument(
        "--feature-extractor-pool-size",
        type=int,
        default=5,
        help="""Number of threads for feature extraction. By default, feature
        extraction are run on CPU.
        """,
    )

    parser.add_argument(
        "--nn-pool-size",
        type=int,
        default=1,
        help="""Number of threads for NN computation and decoding.
        Note: It should be in general less than or equal to num_device
        if num_device is positive.
        """,
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
        from icefall,
        where xxx is the number of BPE tokens you used to train the model.
        Note: Use it only when your model is using BPE. You don't need to
        provide it if you provide `--token-filename`
        """,
    )

    parser.add_argument(
        "--token-filename",
        type=str,
        help="""Filename for tokens.txt
        You can find it in the directory
        egs/aishell/ASR/data/lang_char/tokens.txt from icefall.
        Note: You don't need to provide it if you provide `--bpe-model`
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
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Decoding method to use. Currently, only greedy_search and
        modified_beam_search are implemented.
        """,
    )

    parser.add_argument(
        "--num-active-paths",
        type=int,
        default=4,
        help="""Used only when decoding_method is modified_beam_search.
        It specifies number of active paths for each utterance. Due to
        merging paths with identical token sequences, the actual number
        may be less than "num_active_paths".
        """,
    )

    return parser.parse_args()


@torch.no_grad()
def run_model_and_do_greedy_search(
    model: RnntConformerModel,
    features: List[torch.Tensor],
) -> List[List[int]]:
    """Run RNN-T model with the given features and use greedy search
    to decode the output of the model.

    Args:
      model:
        The RNN-T model.
      features:
        A list of 2-D tensors. Each entry is of shape
        (num_frames, feature_dim).
    Returns:
      Return a list-of-list containing the decoding token IDs.
    """
    features_length = torch.tensor(
        [f.size(0) for f in features],
        dtype=torch.int64,
    )
    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=LOG_EPS,
    )

    device = model.device
    features = features.to(device)
    features_length = features_length.to(device)

    encoder_out, encoder_out_length = model.encoder(
        features=features,
        features_length=features_length,
    )

    hyp_tokens = greedy_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_length=encoder_out_length.cpu(),
    )
    return hyp_tokens


@torch.no_grad()
def run_model_and_do_modified_beam_search(
    model: RnntConformerModel,
    features: List[torch.Tensor],
    num_active_paths: int,
) -> List[List[int]]:
    """Run RNN-T model with the given features and use greedy search
    to decode the output of the model.

    Args:
      model:
        The RNN-T model.
      features:
        A list of 2-D tensors. Each entry is of shape
        (num_frames, feature_dim).
      num_active_paths:
        Used only when decoding_method is modified_beam_search.
        It specifies number of active paths for each utterance. Due to
        merging paths with identical token sequences, the actual number
        may be less than "num_active_paths".
    Returns:
      Return a list-of-list containing the decoding token IDs.
    """
    features_length = torch.tensor(
        [f.size(0) for f in features],
        dtype=torch.int64,
    )
    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=LOG_EPS,
    )

    device = model.device
    features = features.to(device)
    features_length = features_length.to(device)

    encoder_out, encoder_out_length = model.encoder(
        features=features,
        features_length=features_length,
    )

    hyp_tokens = modified_beam_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_length=encoder_out_length.cpu(),
        num_active_paths=num_active_paths,
    )
    return hyp_tokens


class OfflineServer:
    def __init__(
        self,
        nn_model_filename: Optional[str],
        bpe_model_filename: Optional[str],
        token_filename: str,
        num_device: int,
        batch_size: int,
        max_wait_ms: float,
        feature_extractor_pool_size: int,
        nn_pool_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
        decoding_method: str,
        num_active_paths: int,
    ):
        """
        Args:
          nn_model_filename:
            Path to the torch script model.
          bpe_model_filename:
            Path to the BPE model. If it is None, you have to provide
            `token_filename`.
          token_filename:
            Path to tokens.txt. If it is None, you have to provide
            `bpe_model_filename`.
          num_device:
            If 0, use CPU for neural network computation and decoding.
            If positive, it means the number of GPUs to use for NN computation
            and decoding. For each device, there will be a corresponding
            torchscript model. We assume available device IDs are
            0, 1, ... , num_device - 1. You can use the environment variable
            CUDA_VISIBLE_DEVICES to achieve this.
          batch_size:
            Max batch size for inference.
          max_wait_ms:
            Max wait time in milliseconds in order to build a batch of
            `batch_size`.
          feature_extractor_pool_size:
            Number of threads to create for the feature extractor thread pool.
          nn_pool_size:
            Number of threads for the thread pool that is used for NN
            computation and decoding.
          max_message_size:
            Max size in bytes per message.
          max_queue_size:
            Max number of messages in the queue for each connection.
          max_active_connections:
            Max number of active connections. Once number of active client
            equals to this limit, the server refuses to accept new connections.
          decoding_method:
            The decoding method to use. Currently, only greedy_search and
            modified_beam_search are implemented.
          num_active_paths:
            Used only when decoding_method is modified_beam_search.
            It specifies number of active paths for each utterance. Due to
            merging paths with identical token sequences, the actual number
            may be less than "num_active_paths".
        """
        self.feature_extractor = self._build_feature_extractor()
        self.nn_models = self._build_nn_model(nn_model_filename, num_device)

        assert nn_pool_size > 0

        self.feature_extractor_pool = ThreadPoolExecutor(
            max_workers=feature_extractor_pool_size,
            thread_name_prefix="feature",
        )
        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )

        self.feature_queue = asyncio.Queue()

        if bpe_model_filename:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_model_filename)
        else:
            self.token_table = k2.SymbolTable.from_file(token_filename)

        self.counter = 0

        self.max_wait_ms = max_wait_ms
        self.batch_size = batch_size

        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        self.max_active_connections = max_active_connections

        self.current_active_connections = 0

        assert decoding_method in (
            "greedy_search",
            "modified_beam_search",
        ), decoding_method
        if decoding_method == "greedy_search":
            nn_and_decoding_func = run_model_and_do_greedy_search
        elif decoding_method == "modified_beam_search":
            nn_and_decoding_func = functools.partial(
                run_model_and_do_modified_beam_search,
                num_active_paths=num_active_paths,
            )
        else:
            raise ValueError(
                f"Unsupported decoding_method: {decoding_method} "
                "Please use greedy_search or modified_beam_search"
            )

        self.nn_and_decoding_func = nn_and_decoding_func
        self.decoding_method = decoding_method
        self.num_active_paths = num_active_paths

    def _build_feature_extractor(self):
        """Build a fbank feature extractor for extracting features.

        TODO:
          Pass the options as arguments
        """
        opts = kaldifeat.FbankOptions()
        opts.device = "cpu"  # Note: It also supports CUDA, e.g., "cuda:0"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80

        fbank = kaldifeat.Fbank(opts)

        return fbank

    def _build_nn_model(
        self, nn_model_filename: str, num_device: int
    ) -> List[RnntConformerModel]:
        """Build a torch script model for each given device.

        Args:
          nn_model_filename:
            The path to the torch script model.
          num_device:
            Number of devices to use for NN computation and decoding.
            If it is 0, then only use CPU and it returns a model on CPU.
            If it is positive, it create a model for each device and returns
            them.
        Returns:
          Return a list of torch script models.
        """
        if num_device < 1:
            model = RnntConformerModel(
                filename=nn_model_filename,
                device="cpu",
                optimize_for_inference=False,
            )
            return [model]

        ans = []
        for i in range(num_device):
            device = torch.device("cuda", i)
            model = RnntConformerModel(
                filename=nn_model_filename,
                device=device,
                optimize_for_inference=False,
            )
            ans.append(model)

        return ans

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
        logging.info("started")
        task = asyncio.create_task(self.feature_consumer_task())

        # If you use multiple GPUs, you can create multiple
        # feature consumer tasks.
        #  asyncio.create_task(self.feature_consumer_task())
        #  asyncio.create_task(self.feature_consumer_task())
        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=self.max_message_size,
            max_queue=self.max_queue_size,
            process_request=self.process_request,
        ):
            await asyncio.Future()  # run forever
        await task

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Optional[torch.Tensor]:
        """Receives a tensor from the client.

        As the websocket protocol is a message based protocol, not a stream
        protocol, we can receive the whole message sent by the client at once.

        The message from the client is a **bytes** buffer.

        The first message can be either b"Done" meaning the client won't send
        anything in the future or it can be a buffer containing 8 bytes
        in **little** endian format, specifying the number of bytes in the audio
        file, which will be sent by the client in the subsequent messages.
        Since there is a limit in the message size posed by the websocket
        protocol, the client may send the audio file in multiple messages if the
        audio file is very large.

        The second and remaining messages contain audio samples.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a 1-D torch.float32 tensor containing the audio samples or
          return None indicating the end of utterance.
        """
        header = await socket.recv()
        if header == b"Done":
            return None

        assert len(header) == 8, f"The first message should contain 8 bytes"

        expected_num_bytes = int.from_bytes(header, "little", signed=True)

        received = []
        num_received_bytes = 0
        async for message in socket:
            received.append(message)
            num_received_bytes += len(message)

            if num_received_bytes >= expected_num_bytes:
                break

        assert num_received_bytes == expected_num_bytes, (
            num_received_bytes,
            expected_num_bytes,
        )

        samples = b"".join(received)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # PyTorch warns that the underlying buffer is not writable.
            # We ignore it here as we are not going to write it anyway.
            if hasattr(torch, "frombuffer"):
                # Note: torch.frombuffer is available only in torch>= 1.10
                return torch.frombuffer(samples, dtype=torch.float32)
            else:
                array = np.frombuffer(samples, dtype=np.float32)
                return torch.from_numpy(array)

    async def feature_consumer_task(self):
        """This function extracts features from the feature_queue,
        batches them up, sends them to the RNN-T model for computation
        and decoding.
        """
        while True:
            if self.feature_queue.empty():
                await asyncio.sleep(self.max_wait_ms / 1000)
                continue
            batch = []
            try:
                while len(batch) < self.batch_size:
                    item = self.feature_queue.get_nowait()
                    batch.append(item)
            except asyncio.QueueEmpty:
                pass

            feature_list = [b[0] for b in batch]

            loop = asyncio.get_running_loop()
            self.counter = (self.counter + 1) % len(self.nn_models)
            model = self.nn_models[self.counter]

            hyp_tokens = await loop.run_in_executor(
                self.nn_pool,
                self.nn_and_decoding_func,
                model,
                feature_list,
            )

            for i, hyp in enumerate(hyp_tokens):
                self.feature_queue.task_done()
                future = batch[i][1]
                loop.call_soon(future.set_result, hyp)

    async def compute_features(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute the fbank features for the given audio samples.

        Args:
          samples:
            A 1-D torch.float32 tensor containing the audio samples. Its
            sampling rate should be the one as expected by the feature
            extractor. Also, its range should match the one used in the
            training.
        Returns:
          Return a 2-D tensor of shape (num_frames, feature_dim) containing
          the features.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.feature_extractor_pool,
            self.feature_extractor,  # it releases the GIL
            samples,
        )

    async def compute_and_decode(
        self,
        features: torch.Tensor,
    ) -> List[int]:
        """Run the RNN-T model on the features and do greedy search.

        Args:
          features:
            A 2-D tensor of shape (num_frames, feature_dim).
        Returns:
          Return a list of token IDs containing the decoded results.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.feature_queue.put((features, future))
        await future
        return future.result()

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
        """Receive audio samples from the client, process it, and sends
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        logging.info(
            f"Connected: {socket.remote_address}. "
            f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
        )

        while True:
            samples = await self.recv_audio_samples(socket)
            if samples is None:
                break
            features = await self.compute_features(samples)
            hyp = await self.compute_and_decode(features)
            if hasattr(self, "sp"):
                result = self.sp.decode(hyp)
            else:
                result = [self.token_table[i] for i in hyp]
            await socket.send(result)


@torch.no_grad()
def main():
    args = get_args()

    logging.info(vars(args))

    nn_model_filename = args.nn_model_filename
    bpe_model_filename = args.bpe_model_filename
    token_filename = args.token_filename
    port = args.port
    num_device = args.num_device
    max_wait_ms = args.max_wait_ms
    batch_size = args.max_batch_size
    feature_extractor_pool_size = args.feature_extractor_pool_size
    nn_pool_size = args.nn_pool_size
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections
    decoding_method = args.decoding_method
    num_active_paths = args.num_active_paths

    assert decoding_method in ("greedy_search", "modified_beam_search"), decoding_method

    if decoding_method == "modified_beam_search":
        assert num_active_paths >= 1, num_active_paths

    if bpe_model_filename:
        assert token_filename is None

    if token_filename:
        assert bpe_model_filename is None

    offline_server = OfflineServer(
        nn_model_filename=nn_model_filename,
        bpe_model_filename=bpe_model_filename,
        token_filename=token_filename,
        num_device=num_device,
        max_wait_ms=max_wait_ms,
        batch_size=batch_size,
        feature_extractor_pool_size=feature_extractor_pool_size,
        nn_pool_size=nn_pool_size,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
    )
    asyncio.run(offline_server.run(port))


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
    torch.manual_seed(20220519)

    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
