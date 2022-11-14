import argparse
import logging
import os
from typing import Optional, Tuple

import sentencepiece as spm
import torch
from torch import nn

import torch.nn.functional as F
from scaling import ScaledConv1d, ScaledEmbedding, ScaledLinear

from conformer import Conformer
from model import Transducer

from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    display_and_save_batch,
    setup_logger,
    str2bool,
)
from icefall.utils import is_jit_tracing, make_pad_mask

class StreamingEncoder(torch.nn.Module):
    """
    Args:
          left_context:
            How many previous frames the attention can see in current chunk.
            Note: It's not that each individual frame has `left_context` frames
            of left context, some have more.
          right_context:
            How many future frames the attention can see in current chunk.
            Note: It's not that each individual frame has `right_context` frames
            of right context, some have more.
          chunk_size:
            The chunk size for decoding, this will be used to simulate streaming
            decoding using masking.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.
    """

    def __init__(self, model, left_context, right_context, chunk_size, warmup):
        super().__init__()
        self.encoder = model.encoder
        self.encoder_embed = model.encoder_embed
        self.encoder_layers = model.encoder_layers
        self.d_model = model.d_model
        self.cnn_module_kernel = model.cnn_module_kernel
        self.encoder_pos = model.encoder_pos
        self.left_context = left_context
        self.right_context = right_context
        self.chunk_size = chunk_size
        self.warmup = warmup

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        attn_cache: torch.tensor,
        cnn_cache: torch.tensor,
        processed_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states:
            The decode states for previous frames which contains the cached data.
            It has two elements, the first element is the attn_cache which has
            a shape of (encoder_layers, left_context, batch, attention_dim),
            the second element is the conv_cache which has a shape of
            (encoder_layers, cnn_module_kernel-1, batch, conv_dim).
            Note: states will be modified in this function.
          processed_lens:
            How many frames (after subsampling) have been processed for each sequence.

        Returns:
          Return a tuple containing 2 tensors:
            - logits, its shape is (batch_size, output_seq_len, output_dim)
            - logit_lens, a tensor of shape (batch_size,) containing the number
              of frames in `logits` before padding.
            - decode_states, the updated states including the information
              of current chunk.
        """

        # x: [N, T, C]
        # Caution: We assume the subsampling factor is 4!

        #  lengths = ((x_lens - 1) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        lengths = (((x_lens - 1) >> 1) - 1) >> 1
        attn_cache = attn_cache.transpose(0, 2)
        cnn_cache = cnn_cache.transpose(0, 2)
        states = [attn_cache, cnn_cache]
        assert states is not None
        assert processed_lens is not None
        assert (
            len(states) == 2
            and states[0].shape
            == (self.encoder_layers, self.left_context, x.size(0), self.d_model)
            and states[1].shape
            == (
                self.encoder_layers,
                self.cnn_module_kernel - 1,
                x.size(0),
                self.d_model,
            )
        ), f"""The length of states MUST be equal to 2, and the shape of
            first element should be {(self.encoder_layers, self.left_context, x.size(0), self.d_model)},
            given {states[0].shape}. the shape of second element should be
            {(self.encoder_layers, self.cnn_module_kernel - 1, x.size(0), self.d_model)},
            given {states[1].shape}."""

        lengths -= (
            2  # we will cut off 1 frame on each side of encoder_embed output
        )

        embed = self.encoder_embed(x)

        # cut off 1 frame on each size of embed as they see the padding
        # value which causes a training and decoding mismatch.
        embed = embed[:, 1:-1, :]

        embed, pos_enc = self.encoder_pos(embed, self.left_context)
        embed = embed.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)

        src_key_padding_mask = make_pad_mask(lengths, embed.size(0))

        processed_mask = torch.arange(
            self.left_context, device=x.device
        ).expand(x.size(0), self.left_context)
        
        processed_mask = (processed_lens <= processed_mask).flip(1)

        src_key_padding_mask = torch.cat(
            [processed_mask, src_key_padding_mask], dim=1
        )

        x, states = self.encoder.chunk_forward(
            embed,
            pos_enc,
            src_key_padding_mask=src_key_padding_mask,
            warmup=self.warmup,
            states=states,
            left_context=self.left_context,
            right_context=self.right_context,
        )  # (T, B, F)
        if self.right_context > 0:
            x = x[: -self.right_context, ...]
            lengths -= self.right_context

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        processed_lens = processed_lens + lengths.unsqueeze(-1)
        assert processed_lens.shape[1] == 1, processed_lens.shape

        return (
            x,
            lengths,
            states[0].transpose(0, 2),
            states[1].transpose(0, 2),
            processed_lens,
        )

class OfflineEncoder(torch.nn.Module):
    """
    Args:
        model: Conformer Encoder
    """

    def __init__(
        self,
        model
    ) -> None:
        super().__init__()

        self.num_features = model.num_features
        self.subsampling_factor = model.subsampling_factor
        if self.subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = model.encoder_embed

        self.encoder_layers = model.encoder_layers
        self.d_model = model.d_model
        self.cnn_module_kernel = model.cnn_module_kernel
        self.causal = model.causal
        self.dynamic_chunk_training = model.dynamic_chunk_training
        self.short_chunk_threshold = model.short_chunk_threshold
        self.short_chunk_size = model.short_chunk_size
        self.num_left_chunks = model.num_left_chunks

        self.encoder_pos = model.encoder_pos
        self.encoder = model.encoder
       

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, d_model)
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """

        # Note warmup is fixed to 1.0.
        warmup = 1.0
        x = self.encoder_embed(x)
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!

        #  lengths = ((x_lens - 1) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        lengths = (((x_lens - 1) >> 1) - 1) >> 1
        assert max(lengths) < 10000, f"{lengths}, {x_lens}"
        assert min(lengths) > 0,  f"{lengths}, {x_lens}"

        if not is_jit_tracing():
            assert x.size(0) == lengths.max().item()

        src_key_padding_mask = make_pad_mask(lengths, x.size(0))

        if self.dynamic_chunk_training:
            assert (
                self.causal
            ), "Causal convolution is required for streaming conformer."
            max_len = x.size(0)
            chunk_size = torch.randint(1, max_len, (1,)).item()
            if chunk_size > (max_len * self.short_chunk_threshold):
                chunk_size = max_len
            else:
                chunk_size = chunk_size % self.short_chunk_size + 1

            mask = ~subsequent_chunk_mask(
                size=x.size(0),
                chunk_size=chunk_size,
                num_left_chunks=self.num_left_chunks,
                device=x.device,
            )
            x = self.encoder(
                x,
                pos_emb,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                warmup=warmup,
            )  # (T, N, C)
        else:
            x = self.encoder(
                x,
                pos_emb,
                mask=None,
                src_key_padding_mask=src_key_padding_mask,
                warmup=warmup,
            )  # (T, N, C)

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        return x, lengths

class Decoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.

    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.embedding = ScaledEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
            padding_idx=blank_id,
        )
        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = ScaledConv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim,
                bias=False,
            )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        y = y.to(torch.int64)
        embedding_out = self.embedding(y)
        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)

                # During inference time, there is no need to do extra padding
                # as we only need one output
            assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out

class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim)
        self.output_linear = ScaledLinear(joiner_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
           project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        if not is_jit_tracing():
            assert encoder_out.ndim == decoder_out.ndim
            assert encoder_out.ndim in (2, 4)
            assert encoder_out.shape == decoder_out.shape

        
        logit = self.encoder_proj(encoder_out) + self.decoder_proj(
                decoder_out
            )
            
        logit = self.output_linear(torch.tanh(logit))

        return logit

def get_encoder_model(params: AttributeDict) -> nn.Module:
    # TODO: We can add an option to switch between Conformer and Transformer
    encoder = Conformer(
        num_features=params.feature_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.encoder_dim,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        dynamic_chunk_training=params.dynamic_chunk_training,
        short_chunk_size=params.short_chunk_size,
        num_left_chunks=params.num_left_chunks,
        causal=params.causal_convolution,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner

def get_transducer_model(
    params: AttributeDict,
) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return model