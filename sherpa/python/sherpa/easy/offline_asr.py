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
import functools
from typing import List, Optional, Union

import k2
import kaldifeat
import sentencepiece as spm
import torch
from sherpa import RnntConformerModel

from .decode import (
    run_model_and_do_greedy_search,
    run_model_and_do_modified_beam_search,
)


class OfflineAsr(object):
    def __init__(
        self,
        nn_model_filename: str,
        bpe_model_filename: Optional[str],
        token_filename: Optional[str],
        decoding_method: str,
        num_active_paths: int,
        sample_rate: int = 16000,
        device: Union[str, torch.device] = "cpu",
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
          device:
            The device to use for computation.
        """
        self.model = RnntConformerModel(
            filename=nn_model_filename,
            device=device,
            optimize_for_inference=False,
        )

        if bpe_model_filename:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_model_filename)
        else:
            self.token_table = k2.SymbolTable.from_file(token_filename)

        self.feature_extractor = self._build_feature_extractor(
            sample_rate=sample_rate,
            device=device,
        )

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
        self.device = device

    def _build_feature_extractor(
        self,
        sample_rate: int = 16000,
        device: Union[str, torch.device] = "cpu",
    ) -> kaldifeat.OfflineFeature:
        """Build a fbank feature extractor for extracting features.

        Args:
          sample_rate:
            Expected sample rate of the feature extractor.
          device:
            The device to use for computation.
        Returns:
          Return a fbank feature extractor.
        """
        opts = kaldifeat.FbankOptions()
        opts.device = device
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
              Whether it should be in the range [-32768, 32767] or be normalized
              to [-1, 1] depends on which range you used for your training data.
              For instance, if your training data used [-32768, 32767],
              then the given waves have to contain samples in this range.

              All models trained in icefall use the normalized range [-1, 1].
        Returns:
          Return a list of decoded results. `ans[i]` contains the decoded
          results for `wavs[i]`.
        """
        waves = [w.to(self.device) for w in waves]
        features = self.feature_extractor(waves)

        tokens = self.nn_and_decoding_func(self.model, features)

        if hasattr(self, "sp"):
            results = self.sp.decode(tokens)
        else:
            results = [[self.token_table[i] for i in hyp] for hyp in tokens]
            results = ["".join(r) for r in results]

        return results
