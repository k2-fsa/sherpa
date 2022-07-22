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


from typing import List
import math

import torch
from torch.nn.utils.rnn import pad_sequence

from sherpa import greedy_search, modified_beam_search

LOG_EPS = math.log(1e-10)


class GreedySearchOffline:
    def __init__(self, model, device):
        """
        Args:
          model:
            RNN-T model decoder model
          device:
            Device on which the computation will occur
        """

        self.blank_id = model.blank_id
        self.context_size = model.context_size
        self.device = device

        decoder_input = torch.tensor(
            [[self.blank_id] * self.context_size],
            device=self.device,
            dtype=torch.int64,
        )
        initial_decoder_out = model.decoder_forward(decoder_input)
        self.initial_decoder_out = model.forward_decoder_proj(
            initial_decoder_out.squeeze(1)
        )

    @torch.no_grad()
    def process(
        self,
        server: "StreamingServer",
        features: List[torch.Tensor],
    ) -> None:
        """
        Args:
          server:
            An instance of `StreamingServer`.

          features:
            A list of 2-D tensors. Each entry is of shape
            (num_frames, feature_dim).
        Returns:
          Return a list-of-list containing the decoding token IDs.
        """
        model = server.model
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


class ModifiedBeamSearchOffline:
    def __init__(self, blank_id, context_size, num_active_paths):
        self.blank_id = blank_id
        self.context_size = context_size
        self.num_active_paths = num_active_paths

    def process(
        self,
        server: "StreamingServer",
        features: List[torch.Tensor],
    ) -> None:
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
        model = server.model
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
            num_active_paths=self.num_active_paths,
        )
        return hyp_tokens
