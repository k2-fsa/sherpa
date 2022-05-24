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
    ./offline_server.py
"""

import asyncio
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import kaldifeat
import sentencepiece as spm
import torch
import websockets
from _sherpa import RnntModel, greedy_search
from torch.nn.utils.rnn import pad_sequence
import argparse

LOG_EPS = math.log(1e-10)

# You can use
#   icefall/egs/librispeech/ASR/pruned_transducer_statelessX/export.py --jit 1
# to generate the following model
DEFAULT_NN_MODEL_FILENAME = "/ceph-fj/fangjun/open-source-2/icefall-master-2/egs/librispeech/ASR/pruned_transducer_stateless3/exp/cpu_jit.pt"  # noqa
DEFAULT_BPE_MODEL_FILENAME = "/ceph-fj/fangjun/open-source-2/icefall-master-2/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"


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
        even if there are still not enough requests, it still sends the
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

    return parser.parse_args()


def run_model_and_do_greedy_search(
    model: torch.jit.ScriptModule,
    features: List[torch.Tensor],
) -> List[List[int]]:
    """Run RNN-T model with the given features and use greedy search
    to decode the output of the model.

    Args:
      model:
        The RNN-T model.
      features:
        A list of 2-D tensors. Each entry is of shape (num_frames, feature_dim).
    Returns:
      Return a list-of-list containing the decoding token IDs.
    """
    features_length = torch.tensor([f.size(0) for f in features], dtype=torch.int64)
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


class OfflineServer:
    def __init__(
        self,
        nn_model_filename: str,
        bpe_model_filename: str,
        num_device: int,
        batch_size: int,
        max_wait_ms: float,
        feature_extractor_pool_size: int = 3,
        nn_pool_size: int = 3,
    ):
        """
        Args:
          nn_model_filename:
            Path to the torch script model.
          bpe_model_filename:
            Path to the BPE model.
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

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_filename)

        self.counter = 0

        self.max_wait_ms = max_wait_ms
        self.batch_size = batch_size

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
    ) -> List[RnntModel]:
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
            model = RnntModel(
                filename=nn_model_filename,
                device="cpu",
                optimize_for_inference=False,
            )
            return [model]

        ans = []
        for i in range(num_device):
            device = torch.device("cuda", i)
            model = RnntModel(
                filename=nn_model_filename,
                device=device,
                optimize_for_inference=False,
            )
            ans.append(model)

        return ans

    async def loop(self, port: int):
        logging.info("started")
        task = asyncio.create_task(self.feature_consumer_task())

        # If you use multiple GPUs, you can create multiple
        # feature consumer tasks.
        #  asyncio.create_task(self.feature_consumer_task())
        #  asyncio.create_task(self.feature_consumer_task())

        async with websockets.serve(self.handle_connection, "", port):
            await asyncio.Future()  # run forever
        await task

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Optional[torch.Tensor]:
        """Receives a tensor from the client.

        The message from the client has the following format:

            - a header of 8 bytes, containing the number of bytes of the tensor.
              The header is in big endian format.
            - a binary representation of the 1-D torch.float32 tensor.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a 1-D torch.float32 tensor.
        """
        expected_num_bytes = None
        received = b""
        async for message in socket:
            if expected_num_bytes is None:
                assert len(message) >= 8, (len(message), message)
                expected_num_bytes = int.from_bytes(message[:8], "big", signed=True)
                received += message[8:]
                if len(received) == expected_num_bytes:
                    break
            else:
                received += message
                if len(received) == expected_num_bytes:
                    break
        if not received:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # PyTorch warns that the underlying buffer is not writable.
            # We ignore it here as we are not going to write it anyway.
            return torch.frombuffer(received, dtype=torch.float32)

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
                run_model_and_do_greedy_search,
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
        logging.info(f"Connected: {socket.remote_address}")
        while True:
            samples = await self.recv_audio_samples(socket)
            if samples is None:
                break
            features = await self.compute_features(samples)
            hyp = await self.compute_and_decode(features)
            result = self.sp.decode(hyp)
            await socket.send(result)

        logging.info(f"Disconnected: {socket.remote_address}")


@torch.no_grad()
def main():
    args = get_args()

    nn_model_filename = args.nn_model_filename
    bpe_model_filename = args.bpe_model_filename
    port = args.port
    num_device = args.num_device
    max_wait_ms = args.max_wait_ms
    batch_size = args.max_batch_size
    feature_extractor_pool_size = args.feature_extractor_pool_size
    nn_pool_size = args.nn_pool_size

    offline_server = OfflineServer(
        nn_model_filename=nn_model_filename,
        bpe_model_filename=bpe_model_filename,
        num_device=num_device,
        max_wait_ms=max_wait_ms,
        batch_size=batch_size,
        feature_extractor_pool_size=feature_extractor_pool_size,
        nn_pool_size=nn_pool_size,
    )
    asyncio.run(offline_server.loop(port))


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

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
