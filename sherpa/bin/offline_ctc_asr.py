#!/usr/bin/env python3
# noqa
#
# Copyright (c)  2023  Xiaomi Corporation

"""
A standalone script for offline (i.e., non-streaming) speech recognition.

This file decodes files without the need to start a server and a client.

Please refer to
https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc.html
for pre-trained models to download.

Usage:
(1) Use icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
cd icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09

git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe_500/tokens.txt"
git lfs pull --include "data/lang_bpe_500/HLG.pt"

cd /path/to/sherpa

(a) Decoding with H

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt \
  --tokens ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/tokens.txt \
  --use-gpu false \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

(b) Decoding with HLG

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt \
  --tokens ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/tokens.txt \
  --HLG ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt \
  --lm-scale 0.9 \
  --use-gpu false \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

(2) Use wenet-english-model

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/wenet-english-model
cd wenet-english-model
git lfs pull --include "final.zip"

cd /path/to/sherpa

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./wenet-english-model/final.zip \
  --tokens ./wenet-english-model/units.txt \
  --use-gpu false \
  --normalize-samples false \
  ./wenet-english-model/test_wavs/1089-134686-0001.wav \
  ./wenet-english-model/test_wavs/1221-135766-0001.wav \
  ./wenet-english-model/test_wavs/1221-135766-0002.wav

(3) Use wav2vec2.0-torchaudio

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio
cd wav2vec2.0-torchaudio
git lfs pull --include "wav2vec2_asr_base_10m.pt"

cd /path/to/sherpa

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./wav2vec2.0-torchaudio/wav2vec2_asr_base_10m.pt \
  --tokens ./wav2vec2.0-torchaudio/tokens.txt \
  --use-gpu false \
  ./wav2vec2.0-torchaudio/test_wavs/1089-134686-0001.wav \
  ./wav2vec2.0-torchaudio/test_wavs/1221-135766-0001.wav \
  ./wav2vec2.0-torchaudio/test_wavs/1221-135766-0002.wav

(4) Use NeMo CTC models

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-citrinet-512
cd sherpa-nemo-ctc-en-citrinet-512
git lfs pull --include "model.pt"

cd /path/to/sherpa

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./sherpa-nemo-ctc-en-citrinet-512/model.pt
  --tokens ./sherpa-nemo-ctc-en-citrinet-512/tokens.txt \
  --use-gpu false \
  --nemo-normalize per_feature \
  ./sherpa-nemo-ctc-en-citrinet-512/test_wavs/0.wav \
  ./sherpa-nemo-ctc-en-citrinet-512/test_wavs/1.wav \
  ./sherpa-nemo-ctc-en-citrinet-512/test_wavs/2.wav
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
        "--normalize-samples",
        type=str2bool,
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
    if not Path(args.nn_model).is_file():
        raise ValueError(f"{args.nn_model} does not exist")

    if not Path(args.tokens).is_file():
        raise ValueError(f"{args.tokens} does not exist")

    if args.HLG:
        assert Path(args.HLG).is_file(), f"{args.HLG} does not exist"

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


def create_recognizer(args):
    feat_config = sherpa.FeatureConfig()

    feat_config.fbank_opts.frame_opts.samp_freq = 16000
    feat_config.fbank_opts.mel_opts.num_bins = 80
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


def main():
    args = get_parser().parse_args()
    logging.info(vars(args))
    check_args(args)

    recognizer = create_recognizer(args)
    sample_rate = 16000

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
    torch.manual_seed(20230104)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
