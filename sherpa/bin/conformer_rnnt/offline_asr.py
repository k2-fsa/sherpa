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
A standalone script for offline ASR recognition.

It loads a torchscript model, decodes the given wav files, and exits.

Usage:
    ./offline_asr.py --help

For BPE based models (e.g., LibriSpeech):

    ./offline_asr.py \
        --nn-model-filename /path/to/cpu_jit.pt \
        --bpe-model-filename /path/to/bpe.model \
        --decoding-method greedy_search \
        ./foo.wav \
        ./bar.wav \
        ./foobar.wav

For character based models (e.g., aishell):

    ./offline.py \
        --nn-model-filename /path/to/cpu_jit.pt \
        --token-filename /path/to/lang_char/tokens.txt \
        --decoding-method greedy_search \
        ./foo.wav \
        ./bar.wav \
        ./foobar.wav

Note: We provide pre-trained models for testing.

(1) Pre-trained model with the LibriSpeech dataset

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

    nn_model_filename=./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/cpu_jit-torch-1.6.0.pt
    bpe_model=./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/data/lang_bpe_500/bpe.model

    wav1=./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav
    wav2=./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0001.wav
    wav3=./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0002.wav

    sherpa/bin/conformer_rnnt/offline_asr.py \
      --nn-model-filename $nn_model_filename \
      --bpe-model $bpe_model \
      $wav1 \
      $wav2 \
      $wav3

(2) Pre-trained model with the aishell dataset
    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20

    nn_model_filename=./icefall-aishell-pruned-transducer-stateless3-2022-06-20/exp/cpu_jit-epoch-29-avg-5-torch-1.6.0.pt
    token_filename=./icefall-aishell-pruned-transducer-stateless3-2022-06-20/data/lang_char/tokens.txt

    wav1=./icefall-aishell-pruned-transducer-stateless3-2022-06-20/test_wavs/BAC009S0764W0121.wav
    wav2=./icefall-aishell-pruned-transducer-stateless3-2022-06-20/test_wavs/BAC009S0764W0122.wav
    wav3=./icefall-aishell-pruned-transducer-stateless3-2022-06-20/test_wavs/BAC009S0764W0123.wav

    sherpa/bin/conformer_rnnt/offline_asr.py \
      --nn-model-filename $nn_model_filename \
      --token-filename $token_filename \
      $wav1 \
      $wav2 \
      $wav3
"""
import argparse
import functools
import logging
from typing import List, Optional

import k2
import kaldifeat
import sentencepiece as spm
import torch
import torchaudio
from sherpa import RnntConformerModel

from decode import run_model_and_do_greedy_search, run_model_and_do_modified_beam_search


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The expected sample rate of the input sound files",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to equal to `--sample-rate`.",
    )

    return parser.parse_args()


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
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. " f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


class OfflineAsr(object):
    def __init__(
        self,
        nn_model_filename: str,
        bpe_model_filename: Optional[str],
        token_filename: Optional[str],
        decoding_method: str,
        num_active_paths: int,
        sample_rate: float = 16000,
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
          decoding_method:
            The decoding method to use. Currently, only greedy_search and
            modified_beam_search are implemented.
          num_active_paths:
            Used only when decoding_method is modified_beam_search.
            It specifies number of active paths for each utterance. Due to
            merging paths with identical token sequences, the actual number
            may be less than "num_active_paths".
          sample_rate:
            Expected sample rate of the feature extractor.
        """
        self.model = RnntConformerModel(
            filename=nn_model_filename,
            device="cpu",
            optimize_for_inference=False,
        )

        if bpe_model_filename:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_model_filename)
        else:
            self.token_table = k2.SymbolTable.from_file(token_filename)

        self.feature_extractor = self._build_feature_extractor(sample_rate)

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

    def _build_feature_extractor(
        self,
        sample_rate: float = 16000,
    ) -> kaldifeat.OfflineFeature:
        """Build a fbank feature extractor for extracting features.

        Args:
          sample_rate:
            Expected sample rate of the feature extractor.

        """
        opts = kaldifeat.FbankOptions()
        opts.device = "cpu"  # Note: It also supports CUDA, e.g., "cuda:0"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = sample_rate
        opts.mel_opts.num_bins = 80

        fbank = kaldifeat.Fbank(opts)

        return fbank

    def decode_waves(self, waves: List[torch.Tensor]) -> List[List[str]]:
        """
        Args:
          waves:
            A list of 1-D torch.float32 tensors containing audio samples.
            wavs[i] contains audio samples for the i-th utterance.

            Note:
              Whether it should be in the range [-32768, 32676] or be normalized
              to [-1, 1] depends on which range you used for your training data.

              All models trained by icefall use normalized range [-1, 1].
        Returns:
          Return a list of decoded results. `ans[i]` contains the decoded
          results for `wavs[i]`.
        """
        features = self.feature_extractor(waves)

        tokens = self.nn_and_decoding_func(self.model, features)

        if hasattr(self, "sp"):
            results = self.sp.decode(tokens)
        else:
            results = [[self.token_table[i] for i in hyp] for hyp in tokens]
            results = ["".join(r) for r in results]

        return results


@torch.no_grad()
def main():
    args = get_args()
    logging.info(vars(args))

    nn_model_filename = args.nn_model_filename
    bpe_model_filename = args.bpe_model_filename
    token_filename = args.token_filename
    decoding_method = args.decoding_method
    num_active_paths = args.num_active_paths
    sample_rate = args.sample_rate
    sound_files = args.sound_files

    assert decoding_method in ("greedy_search", "modified_beam_search"), decoding_method

    if decoding_method == "modified_beam_search":
        assert num_active_paths >= 1, num_active_paths

    if bpe_model_filename:
        assert token_filename is None

    if token_filename:
        assert bpe_model_filename is None

    offline_asr = OfflineAsr(
        nn_model_filename=nn_model_filename,
        bpe_model_filename=bpe_model_filename,
        token_filename=token_filename,
        decoding_method=decoding_method,
        num_active_paths=num_active_paths,
        sample_rate=sample_rate,
    )

    waves = read_sound_files(
        filenames=sound_files,
        expected_sample_rate=sample_rate,
    )

    hyps = offline_asr.decode_waves(waves)

    s = "\n"
    for filename, hyp in zip(sound_files, hyps):
        s += f"{filename}:\n{hyp}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    torch.manual_seed(20220609)

    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
