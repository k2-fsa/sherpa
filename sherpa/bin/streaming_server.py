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

import logging
import argparse
import asyncio
from typing import List, Optional, Tuple

import sentencepiece as spm
import torch
import torchaudio
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature

DEFAULT_NN_MODEL_FILENAME = "/ceph-fj/fangjun/open-source-2/icefall-streaming-2/egs/librispeech/ASR/pruned_stateless_emformer_rnnt2/exp-full/cpu_jit-epoch-39-avg-6-use-averaged-model-1.pt"  # noqa
DEFAULT_BPE_MODEL_FILENAME = "/ceph-fj/fangjun/open-source-2/icefall-streaming-2/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"  # noqa

TEST_WAV = "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav"
TEST_WAV = "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0001.wav"
TEST_WAV = "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0002.wav"


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


def greedy_search(
    model: torch.jit.ScriptModule,
    encoder_out: torch.Tensor,
    hyp: List[int],
    sp: spm.SentencePieceProcessor,
    decoder_out: Optional[torch.Tensor] = None,
):
    """
    Args:
      model:
        The RNN-T model.
      encoder_out:
        A 3-D tensor of shape (N, T, encoder_out_dim) containing the output of
        the encoder model.
      sp:
        The BPE model.
    """
    assert encoder_out.ndim == 3

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device
    T = encoder_out.size(1)

    if decoder_out is None:
        decoder_input = torch.tensor(
            [hyp[-context_size:]],
            device=device,
            dtype=torch.int64,
        )

        decoder_out = model.decoder(decoder_input, need_pad=False).squeeze(1)
        # decoder_out is of shape (N, decoder_out_dim)

    for t in range(T):
        current_encoder_out = encoder_out[:, t]
        # current_encoder_out's shape: (batch_size, encoder_out_dim)

        logits = model.joiner(current_encoder_out, decoder_out)
        # logits'shape (batch_size,  vocab_size)

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                hyp.append(v)
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = torch.tensor(
                [hyp[-context_size:]],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(
                decoder_input,
                need_pad=False,
            ).squeeze(1)

            logging.info(f"Partial result :\n{sp.decode(hyp[context_size:])}")

    return hyp, decoder_out


def process_features(
    model: torch.jit.ScriptModule,
    features: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    hyp: List[int],
    decoder_out: Optional[torch.Tensor] = None,
    state: Optional[List[List[torch.Tensor]]] = None,
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

    device = next(model.parameters()).device
    features = features.to(device)
    feature_lens = torch.full(
        (batch_size,),
        fill_value=features.size(1),
        device=device,
    )

    states = model.encoder.get_init_state()

    (encoder_out, encoder_out_lens, state) = model.encoder.streaming_forward(
        features,
        feature_lens,
        state,
    )

    hyp, decoder_out = greedy_search(
        model=model,
        encoder_out=encoder_out,
        hyp=hyp,
        sp=sp,
        decoder_out=decoder_out,
    )

    return hyp, decoder_out, state


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

        self.model = torch.jit.load(nn_model_filename, map_location=device)

        # number of frames before subsampling
        self.segment_length = self.model.encoder.segment_length

        self.right_context_length = self.model.encoder.right_context_length

        # We add 3 here since the subsampling method is using
        # ((len - 1) // 2 - 1) // 2)
        self.chunk_length = (self.segment_length + 3) + self.right_context_length

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_filename)

        self.context_size = self.model.decoder.context_size
        self.blank_id = self.model.decoder.blank_id
        self.log_eps = self.model.encoder.log_eps

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
        decoder_out = None
        state = None
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
                hyp, decoder_out, state = process_features(
                    model=self.model,
                    features=chunk,
                    sp=self.sp,
                    hyp=hyp,
                    decoder_out=decoder_out,
                    state=state,
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
            hyp=hyp,
            decoder_out=decoder_out,
            state=state,
        )
        logging.info(f"Final results: {self.sp.decode(hyp[self.context_size:])}")


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
