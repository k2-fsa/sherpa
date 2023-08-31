#!/usr/bin/env python3
# Copyright      2022-2023  Xiaomi Corp.
"""
A server for transducer-based offline ASR. Offline means you send all
the content of the audio for recognition.

It supports multiple clients sending at the same time.

Usage:
    ./offline_transducer_server.py --help

    ./offline_transducer_server.py

Please refer to
https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_transducer.html
for pre-trained models to download.

We use the Zipformer pre-trained model below to demonstrate how to use
this file. You can use other non-streaming transducer models with this file
if you want.

(1) Download pre-trained models

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
cd icefall-asr-librispeech-zipformer-2023-05-15

git lfs pull --include "exp/jit_script.pt"

(2) Start the server

cd /path/to/sherpa

./sherpa/bin/offline_transducer_server.py \
  --port 6006 \
  --nn-model ./icefall-asr-librispeech-zipformer-2023-05-15/exp/jit_script.pt \
  --tokens ./icefall-asr-librispeech-zipformer-2023-05-15/data/lang_bpe_500/tokens.txt

(3) Start the client

python3 ./sherpa/bin/offline_client.py ./icefall-asr-librispeech-zipformer-2023-05-15/test_wavs/1089-134686-0001.wav
"""  # noqa

import argparse
import asyncio
import http
import logging
import socket
import ssl
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import websockets

import sherpa


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--nn-model",
        type=str,
        help="""The torchscript model. Please refer to
        https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_transducer.html
        for a list of pre-trained models to download.
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the data used to train the model. ",
    )

    parser.add_argument(
        "--feat-dim",
        type=int,
        default=80,
        help="Feature dimension of the model",
    )

    parser.add_argument(
        "--use-bbpe",
        type=sherpa.str2bool,
        default=False,
        help="Whether the model to be used is trained with bbpe",
    )


def add_decoding_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Decoding method to use. Current supported methods are:
        - greedy_search
        - modified_beam_search
        - fast_beam_search
        """,
    )

    add_modified_beam_search_args(parser)
    add_fast_beam_search_args(parser)


def add_modified_beam_search_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-active-paths",
        type=int,
        default=4,
        help="""Used only when --decoding-method is modified_beam_search.
        It specifies number of active paths to keep during decoding.
        """,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="""Used only when --decoding-method is modified_beam_search.
        It specifies the softmax temperature.
        """,
    )


def add_fast_beam_search_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="Used only when --decoding-method is fast_beam_search",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="Used only when --decoding-method is fast_beam_search",
    )

    parser.add_argument(
        "--allow-partial",
        type=sherpa.str2bool,
        default=True,
        help="Used only when --decoding-method is fast_beam_search",
    )

    parser.add_argument(
        "--LG",
        type=str,
        default="",
        help="""Used only when --decoding-method is fast_beam_search.
        If not empty, it points to LG.pt.
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search and
        --LG is not empty.
        """,
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --method is fast_beam_search""",
    )


def add_resources_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use-gpu",
        type=sherpa.str2bool,
        default=False,
        help="""True to use GPU. It always selects GPU 0. You can use the
        environement variable CUDA_VISIBLE_DEVICES to control which GPU
        is mapped to GPU 0.
        """,
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Sets the number of threads used for interop parallelism "
        "(e.g. in JIT interpreter) on CPU.",
    )


def check_args(args):
    if args.use_gpu and not torch.cuda.is_available():
        sys.exit("no CUDA devices available but you set --use-gpu=true")

    if not Path(args.nn_model).is_file():
        raise ValueError(f"{args.nn_model} does not exist")

    if not Path(args.tokens).is_file():
        raise ValueError(f"{args.tokens} does not exist")

    if args.decoding_method not in (
        "greedy_search",
        "modified_beam_search",
        "fast_beam_search",
    ):
        raise ValueError(f"Unsupported decoding method {args.decoding_method}")

    if args.decoding_method == "modified_beam_search":
        assert args.num_active_paths > 0, args.num_active_paths
        assert args.temperature > 0, args.temperature

    if args.decoding_method == "fast_beam_search" and args.LG:
        if not Path(args.LG).is_file():
            raise ValueError(f"{args.LG} does not exist")


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_model_args(parser)
    add_decoding_args(parser)
    add_resources_args(parser)

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="The server will listen on this port",
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
        extraction runs on CPU.
        """,
    )

    parser.add_argument(
        "--nn-pool-size",
        type=int,
        default=1,
        help="Number of threads for NN computation and decoding.",
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
        "--certificate",
        type=str,
        help="""Path to the X.509 certificate. You need it only if you want to
        use a secure websocket connection, i.e., use wss:// instead of ws://.
        You can use sherpa/bin/web/generate-certificate.py
        to generate the certificate `cert.pem`.
        """,
    )

    parser.add_argument(
        "--doc-root",
        type=str,
        default="./sherpa/bin/web",
        help="""Path to the web root""",
    )

    return parser.parse_args()


class OfflineServer:
    def __init__(
        self,
        recognizer: sherpa.OfflineRecognizer,
        max_batch_size: int,
        max_wait_ms: float,
        feature_extractor_pool_size: int,
        nn_pool_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
        doc_root: str,
        certificate: Optional[str] = None,
    ):
        """
        Args:
          recognizer:
            An instance of the sherpa.OfflineRecognizer.
          max_batch_size:
            Max batch size for inference.
          max_wait_ms:
            Max wait time in milliseconds in order to build a batch of
            `max_batch_size`.
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
          doc_root:
            Path to the directory where files like index.html for the HTTP
            server locate.
          certificate:
            Optional. If not None, it will use secure websocket.
            You can use ./sherpa/bin/web/generate-certificate.py to generate
            it (the default generated filename is `cert.pem`).
        """
        self.recognizer = recognizer

        self.certificate = certificate
        self.http_server = sherpa.HttpServer(doc_root)

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

    async def process_request(
        self,
        path: str,
        request_headers: websockets.Headers,
    ) -> Optional[Tuple[http.HTTPStatus, websockets.Headers, bytes]]:
        if "sec-websocket-key" not in request_headers:
            # This is a normal HTTP request
            if path == "/":
                path = "/index.html"
            found, response, mime_type = self.http_server.process_request(path)
            if isinstance(response, str):
                response = response.encode("utf-8")

            if not found:
                status = http.HTTPStatus.NOT_FOUND
            else:
                status = http.HTTPStatus.OK
            header = {"Content-Type": mime_type}
            return status, header, response

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

        task = asyncio.create_task(self.stream_consumer_task())

        if self.certificate:
            logging.info(f"Using certificate: {self.certificate}")
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.certificate)
        else:
            ssl_context = None
            logging.info("No certificate provided")

        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=self.max_message_size,
            max_queue=self.max_queue_size,
            process_request=self.process_request,
            ssl=ssl_context,
        ):
            ip_list = ["0.0.0.0", "localhost", "127.0.0.1"]
            ip_list.append(socket.gethostbyname(socket.gethostname()))
            proto = "http://" if ssl_context is None else "https://"
            s = "Please visit one of the following addresses:\n\n"
            for p in ip_list:
                s += "  " + proto + p + f":{port}" "\n"
            logging.info(s)

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

        The first message can be either "Done" meaning the client won't send
        anything in the future or it can be a buffer containing 4 bytes
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
        if header == "Done":
            return None

        assert len(header) == 4, "The first message should contain 4 bytes"

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

                    batch.append(item)
            except asyncio.QueueEmpty:
                pass
            stream_list = [b[0] for b in batch]
            future_list = [b[1] for b in batch]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.nn_pool,
                self.recognizer.decode_streams,
                stream_list,
            )

            for f in future_list:
                self.stream_queue.task_done()
                f.set_result(None)

    async def compute_and_decode(
        self,
        stream: sherpa.OfflineStream,
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
        except websockets.exceptions.ConnectionClosedError:
            logging.info(f"{socket.remote_address} disconnected")
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

        while True:
            stream = self.recognizer.create_stream()
            samples = await self.recv_audio_samples(socket)
            if samples is None:
                break
            # stream.accept_samples() runs in the main thread
            # TODO(fangjun): Use a separate thread/process pool for it
            stream.accept_samples(samples)

            await self.compute_and_decode(stream)
            result = stream.result.text
            logging.info(f"result: {result}")

            if result:
                await socket.send(result)
            else:
                # If result is an empty string, send something to the client.
                # Otherwise, socket.send() is a no-op and the client will
                # wait for a reply indefinitely.
                await socket.send("<EMPTY>")


def create_recognizer(args) -> sherpa.OfflineRecognizer:
    feat_config = sherpa.FeatureConfig()

    feat_config.fbank_opts.frame_opts.samp_freq = args.sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = args.feat_dim
    feat_config.fbank_opts.frame_opts.dither = 0

    fast_beam_search_config = sherpa.FastBeamSearchConfig(
        lg=args.LG if args.LG else "",
        ngram_lm_scale=args.ngram_lm_scale,
        beam=args.beam,
        max_states=args.max_states,
        max_contexts=args.max_contexts,
        allow_partial=args.allow_partial,
    )

    config = sherpa.OfflineRecognizerConfig(
        nn_model=args.nn_model,
        tokens=args.tokens,
        use_gpu=args.use_gpu,
        num_active_paths=args.num_active_paths,
        use_bbpe=args.use_bbpe,
        feat_config=feat_config,
        decoding_method=args.decoding_method,
        fast_beam_search_config=fast_beam_search_config,
        temperature=args.temperature
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


@torch.no_grad()
def main():
    args = get_args()
    logging.info(vars(args))
    check_args(args)

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)
    recognizer = create_recognizer(args)

    port = args.port
    max_wait_ms = args.max_wait_ms
    max_batch_size = args.max_batch_size
    feature_extractor_pool_size = args.feature_extractor_pool_size
    nn_pool_size = args.nn_pool_size
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections
    certificate = args.certificate
    doc_root = args.doc_root

    if certificate and not Path(certificate).is_file():
        raise ValueError(f"{certificate} does not exist")

    if not Path(doc_root).is_dir():
        raise ValueError(f"Directory {doc_root} does not exist")

    offline_server = OfflineServer(
        recognizer=recognizer,
        max_wait_ms=max_wait_ms,
        max_batch_size=max_batch_size,
        feature_extractor_pool_size=feature_extractor_pool_size,
        nn_pool_size=nn_pool_size,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
        certificate=certificate,
        doc_root=doc_root,
    )
    asyncio.run(offline_server.run(port))


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
    log_filename = "log/log-offline-transducer-server"
    sherpa.setup_logger(log_filename)
    main()
