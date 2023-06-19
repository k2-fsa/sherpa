#!/usr/bin/env python3
# Copyright      2022-2023  Xiaomi Corp.
"""
A server for offline CTC ASR recognition. Offline means you send all the content
of the audio for recognition.

It supports multiple clients sending at the same time.

Usage:
    ./offline_ctc_server.py --help

    ./offline_ctc_server.py

Please refer to
https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc/index.html
for pre-trained models to download.

We use a Conformer CTC pre-trained model from NeMO below to demonstrate how to use
this file. You can use other non-streaming CTC models with this file
if you want.

(1) Download pre-trained models

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-conformer-medium
cd sherpa-nemo-ctc-en-conformer-medium
git lfs pull --include "model.pt"

(2) Start the server

cd /path/to/sherpa

./sherpa/bin/offline_ctc_server.py \
  --port 6006 \
  --nemo-normalize=per_feature \
  --nn-model ./sherpa-nemo-ctc-en-conformer-medium/model.pt \
  --tokens ./sherpa-nemo-ctc-en-conformer-medium/tokens.txt \

(3) Start the client

python3 ./sherpa/bin/offline_client.py ./sherpa-nemo-ctc-en-conformer-medium/test_wavs/0.wav
"""  # noqa

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import torch
from offline_transducer_server import OfflineServer, add_resources_args

import sherpa


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--nn-model",
        type=str,
        help="""The torchscript model. Please refer to
        https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc/index.html
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
        "--normalize-samples",
        type=sherpa.str2bool,
        default=True,
        help="""If your model was trained using features computed
        from samples in the range `[-32768, 32767]`, then please set
        this flag to False. For instance, if you use models from WeNet,
        please set it to False.
        """,
    )

    parser.add_argument(
        "--nemo-normalize",
        type=str,
        default="",
        help="""Used only for models from NeMo.
        Leave it to empty if the preprocessor of the model does not use
        normalization. Current supported value is "per_feature".
        """,
    )


def add_decoding_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--HLG",
        type=str,
        help="""Optional. If empty, we use an H graph for decoding.
        If not empty, it is the filename of HLG.pt and we will
        use it for decoding""",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=1.0,
        help="""
        Used only when --HLG is not empty. It specifies the scale
        for HLG.scores
        """,
    )

    parser.add_argument(
        "--modified",
        type=bool,
        default=True,
        help="""Used only when --HLG is empty. True to use a modified
        CTC topology. False to use a standard CTC topology.
        Please refer to https://k2-fsa.github.io/k2/python_api/api.html#ctc-topo
        for the differences between standard and modified CTC topology.
        """,
    )

    parser.add_argument(
        "--search-beam",
        type=float,
        default=20.0,
        help="""Decoding beam, e.g. 20.  Smaller is faster, larger is
        more exact (less pruning). This is the default value;
        it may be modified by `min_active_states` and
        `max_active_states`.
        """,
    )

    parser.add_argument(
        "--output-beam",
        type=float,
        default=8.0,
        help="""Beam to prune output, similar to lattice-beam in Kaldi.
        Relative to the best path of output.
        """,
    )

    parser.add_argument(
        "--min-active-states",
        type=int,
        default=30,
        help="""Minimum number of FSA states that are allowed to
         be active on any given frame for any given
        intersection/composition task. This is advisory,
        in that it will try not to have fewer than this
        number active. Set it to zero if there is no
        constraint.""",
    )

    parser.add_argument(
        "--max-active-states",
        type=int,
        default=10000,
        help="""Maximum number of FSA states that are allowed to
        be active on any given frame for any given
        intersection/composition task. This is advisory,
        in that it will try not to exceed that but may
        not always succeed. You can use a very large
        number if no constraint is needed.""",
    )


def check_args(args):
    if args.use_gpu and not torch.cuda.is_available():
        sys.exit("no CUDA devices available but you set --use-gpu=true")

    if not Path(args.nn_model).is_file():
        raise ValueError(f"{args.nn_model} does not exist")

    if not Path(args.tokens).is_file():
        raise ValueError(f"{args.tokens} does not exist")

    if args.HLG:
        assert Path(args.HLG).is_file(), f"{args.HLG} does not exist"


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


def create_recognizer(args) -> sherpa.OfflineRecognizer:
    feat_config = sherpa.FeatureConfig()

    feat_config.fbank_opts.frame_opts.samp_freq = args.sample_rate
    feat_config.fbank_opts.mel_opts.num_bins = args.feat_dim
    feat_config.fbank_opts.frame_opts.dither = 0

    feat_config.normalize_samples = args.normalize_samples
    feat_config.nemo_normalize = args.nemo_normalize

    ctc_decoder_config = sherpa.OfflineCtcDecoderConfig(
        hlg=args.HLG if args.HLG else "",
        lm_scale=args.lm_scale,
        modified=args.modified,
        search_beam=args.search_beam,
        output_beam=args.output_beam,
        min_active_states=args.min_active_states,
        max_active_states=args.max_active_states,
    )

    config = sherpa.OfflineRecognizerConfig(
        nn_model=args.nn_model,
        tokens=args.tokens,
        use_gpu=args.use_gpu,
        feat_config=feat_config,
        ctc_decoder_config=ctc_decoder_config,
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
    log_filename = "log/log-offline-ctc-server"
    sherpa.setup_logger(log_filename)
    main()
