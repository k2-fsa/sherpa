#!/usr/bin/env python3
# noqa
#
# Copyright (c)  2023  Xiaomi Corporation

"""
A standalone script for offline (i.e., non-streaming) speech recognition.

This file decodes files without the need to start a server and a client.

Please refer to
https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_transducer.html#
for pre-trained models to download.

See
https://k2-fsa.github.io/sherpa/python/offline_asr/standalone/transducer.html
for detailed usages and also you can find a colab notebook there.

We use the Zipformer pre-trained model below to demonstrate how to use
this file:

(1) Download pre-trained models

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02
cd icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02
git lfs pull --include "exp/cpu_jit-torch-1.10.pt"
git lfs pull --include "data/lang_bpe_500/LG.pt"

(2) greedy_search

cd /path/to/sherpa

./sherpa/bin/offline_transducer_asr.py \
  --nn-model ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/exp/cpu_jit-torch-1.10.pt \
  --tokens ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/tokens.txt \
  --decoding-method greedy_search \
  --use-gpu false \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0002.wav

(3) modified_beam_search

cd /path/to/sherpa

./sherpa/bin/offline_transducer_asr.py \
  --nn-model ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/exp/cpu_jit-torch-1.10.pt \
  --tokens ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/tokens.txt \
  --decoding-method modified_beam_search \
  --num-active-paths 4 \
  --use-gpu false \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0002.wav

(4) fast_beam_search (without LG)

cd /path/to/sherpa

./sherpa/bin/offline_transducer_asr.py \
  --nn-model ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/exp/cpu_jit-torch-1.10.pt \
  --tokens ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/tokens.txt \
  --decoding-method fast_beam_search \
  --max-contexts 8 \
  --max-states 64 \
  --allow-partial true \
  --beam 4 \
  --use-gpu false \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0002.wav

(5) fast_beam_search (with LG)

cd /path/to/sherpa

./sherpa/bin/offline_transducer_asr.py \
  --nn-model ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/exp/cpu_jit-torch-1.10.pt \
  --tokens ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/tokens.txt \
  --decoding-method fast_beam_search \
  --max-contexts 8 \
  --max-states 64 \
  --allow-partial true \
  --beam 4 \
  --LG ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/data/lang_bpe_500/LG.pt \
  --ngram-lm-scale 0.01 \
  --use-gpu false \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02/test_wavs/1221-135766-0002.wav
"""
import argparse
import logging
from pathlib import Path
from typing import List

import torch
import torchaudio

import sherpa
from sherpa import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_model_args(parser)
    add_decoding_args(parser)
    add_resources_args(parser)

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. ",
    )

    return parser


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--nn-model",
        type=str,
        help="""The torchscript model. Please refer to
        https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc.html
        and
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
        help="Sample rate of the data used to train the model. "
        "Caution: If your input sound files have a different sampling rate, "
        "we will do resampling inside",
    )

    parser.add_argument(
        "--feat-dim",
        type=int,
        default=80,
        help="Feature dimension of the model",
    )


def add_decoding_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--decoding-method",
        type=str,
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
        type=str2bool,
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
        type=str2bool,
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
        help="Sets the number of threads used for interop parallelism (e.g. in JIT interpreter) on CPU.",
    )


def check_args(args):
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

    if args.decoding_method == "fast_beam_search" and args.LG:
        if not Path(args.LG).is_file():
            raise ValueError(f"{args.LG} does not exist")

    assert len(args.sound_files) > 0, args.sound_files
    for f in args.sound_files:
        if not Path(f).is_file():
            raise ValueError(f"{f} does not exist")


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        if sample_rate != expected_sample_rate:
            wave = torchaudio.functional.resample(
                wave,
                orig_freq=sample_rate,
                new_freq=expected_sample_rate,
            )

        # We use only the first channel
        ans.append(wave[0].contiguous())
    return ans


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
        feat_config=feat_config,
        decoding_method=args.decoding_method,
        fast_beam_search_config=fast_beam_search_config,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


def main():
    args = get_parser().parse_args()
    logging.info(vars(args))
    check_args(args)

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)

    recognizer = create_recognizer(args)
    sample_rate = args.sample_rate

    samples: List[torch.Tensor] = read_sound_files(
        args.sound_files,
        sample_rate,
    )

    streams: List[sherpa.OfflineStream] = []
    for s in samples:
        stream = recognizer.create_stream()
        stream.accept_samples(s)
        streams.append(stream)

    recognizer.decode_streams(streams)
    for filename, stream in zip(args.sound_files, streams):
        print(f"{filename}\n{stream.result}")


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
    torch.manual_seed(20230104)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
else:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
