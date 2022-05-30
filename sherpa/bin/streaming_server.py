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
from typing import List, Optional, Tuple

import sentencepiece as spm
import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature

from sherpa import RnntEmformerModel, streaming_greedy_search

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


def process_features(
    model: RnntEmformerModel,
    features: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    hyps: List[int],
    decoder_out: torch.Tensor,
    states: List[List[torch.Tensor]],
) -> Tuple[List[int], torch.Tensor, List[List[torch.Tensor]]]:
    """Process features for each stream in parallel.

    Args:
      model:
        The RNN-T model.
      features:
        A 3-D tensor of shape (N, T, C).
      sp:
        Then BPE model.
    """
    assert features.ndim == 3
    batch_size = features.size(0)

    device = model.device
    features = features.to(device)
    feature_lens = torch.full(
        (batch_size,),
        fill_value=features.size(1),
        device=device,
    )

    (encoder_out, states) = model.encoder_streaming_forward(
        features,
        feature_lens,
        states,
    )
    decoder_out, hyps = streaming_greedy_search(
        model=model, encoder_out=encoder_out, decoder_out=decoder_out, hyps=[hyps]
    )
    logging.info(f"Partial result: {sp.decode(hyps[0][2:])}")

    return hyps[0], decoder_out, states


class StreamingServer(object):
    def __init__(
        self,
        nn_model_filename: str,
        bpe_model_filename: str,
    ):
        """
        Args:
          nn_model_filename:
            Path to the torchscript model
          bpe_model_filename:
            Path to the BPE model
        """
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")

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

    def _create_streaming_feature_extractor(self) -> OnlineFeature:
        """Create a CPU streaming feature extractor.

        At present, we assume it returns a fbank feature extractor with
        fixed options. In the future, we will support passing in the options
        from outside.

        Returns:
          Return a CPU streaming feature extractor.
        """
        opts = FbankOptions()
        opts.device = "cpu"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        return OnlineFbank(opts)

    def loop(self):
        feat_extractor = self._create_streaming_feature_extractor()
        chunk_size = 1024

        wav, sample_rate = torchaudio.load(TEST_WAV)
        wav = wav.squeeze(0)
        features = []
        num_processed_frames = 0
        start = 0
        hyp = [self.blank_id] * self.context_size

        decoder_input = torch.tensor(
            [hyp],
            device=self.model.device,
            dtype=torch.int64,
        )

        decoder_out = self.model.decoder_forward(decoder_input).squeeze(1)
        states = self.model.get_encoder_init_state()
        while True:
            end = start + chunk_size
            data = wav[start:end]
            start = end
            if data.numel() == 0:
                break
            feat_extractor.accept_waveform(
                sampling_rate=sample_rate,
                waveform=data,
            )

            while num_processed_frames < feat_extractor.num_frames_ready:
                f = feat_extractor.get_frame(num_processed_frames)
                features.append(f)
                num_processed_frames += 1

            if len(features) >= self.chunk_length:
                chunk = features[: self.chunk_length]
                features = features[self.segment_length :]
                chunk = torch.cat(chunk, dim=0)
                chunk = chunk.unsqueeze(0)
                hyp, decoder_out, states = process_features(
                    model=self.model,
                    features=chunk,
                    sp=self.sp,
                    hyps=hyp,
                    decoder_out=decoder_out,
                    states=states,
                )

        feat_extractor.input_finished()

        while num_processed_frames < feat_extractor.num_frames_ready:
            f = feat_extractor.get_frame(num_processed_frames)
            features.append(f)
            num_processed_frames += 1

        chunk = torch.cat(features, dim=0)
        chunk = torch.nn.functional.pad(
            chunk,
            (0, 0, 0, self.chunk_length - chunk.size(0)),
            mode="constant",
            value=self.log_eps,
        )
        chunk = chunk.unsqueeze(0)
        hyp, decoder_out, state = process_features(
            model=self.model,
            features=chunk,
            sp=self.sp,
            hyps=hyp,
            decoder_out=decoder_out,
            states=states,
        )
        logging.info(f"Final results: {self.sp.decode(hyp[self.context_size:])}")


@torch.no_grad()
def main():
    args = get_args()
    nn_model_filename = args.nn_model_filename
    bpe_model_filename = args.bpe_model_filename

    server = StreamingServer(
        nn_model_filename=nn_model_filename,
        bpe_model_filename=bpe_model_filename,
    )
    server.loop()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
