#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
# flake8: noqa

"""
Note: Code in this file is modified from
https://github.com/TadaoYamaoka/whisper/blob/main/to_onnx.py

Thanks to https://github.com/TadaoYamaoka
for making the onnx export script public.

Note that we have removed the 30 seconds constraint from whisper. You can
use any T <= 30.
"""

"""
tiny.en:
ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4, n_vocab=51864, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4)
"""

import argparse
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import whisper
from whisper.model import (
    AudioEncoder,
    Conv1d,
    LayerNorm,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # fmt: off
        choices=[
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en",
            "large-v1", "large-v2",
            "large", "large-v3", "turbo", # these three have feature dim 128
            "distil-medium.en", "distil-small.en", "distil-large-v2",
            # "distil-large-v3", # distil-large-v3 is not supported!
            # for fine-tuned models from icefall
            "medium-aishell",
            ],
        # fmt: on
    )
    return parser.parse_args()


# Copied from https://pytorch.org/docs/1.9.0/_modules/torch/nn/modules/module.html#Module.get_submodule  # noqa
# get_submodule was added to nn.Module at v1.9.0
def get_submodule(model, target):
    if target == "":
        return model
    atoms: List[str] = target.split(".")
    mod: torch.nn.Module = model
    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(
                mod._get_name() + " has no " "attribute `" + item + "`"
            )
        mod = getattr(mod, item)
        if not isinstance(mod, torch.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")
    return mod


class ModifiedConv1d(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv = nn.Conv1d(
            m.in_channels,
            m.out_channels,
            kernel_size=m.kernel_size,
            padding=m.padding,
            stride=m.stride,
        )
        self.conv.weight = m.weight
        self.conv.bias = m.bias

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ModifiedLayerNorm(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.layer = nn.LayerNorm(m.normalized_shape)
        self.layer.weight = m.weight
        self.layer.bias = m.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class AudioEncoderTensorCache(nn.Module):
    def __init__(self, inAudioEncoder: AudioEncoder, inTextDecoder: TextDecoder):
        super().__init__()
        self.audioEncoder = inAudioEncoder
        self.textDecoder = inTextDecoder

    def forward(self, x: Tensor):
        audio_features = self.audioEncoder(x)

        n_layer_cross_k_list: List[torch.Tensor] = []
        n_layer_cross_v_list: List[torch.Tensor] = []
        for block in self.textDecoder.blocks:
            n_layer_cross_k_list.append(block.cross_attn.key(audio_features))
            n_layer_cross_v_list.append(block.cross_attn.value(audio_features))

        return torch.stack(n_layer_cross_k_list), torch.stack(n_layer_cross_v_list)


class MultiHeadAttentionCross(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.multiHeadAttention.query(x)

        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.multiHeadAttention.n_head) ** -0.25
        q = (
            q.view(q.shape[0], q.shape[1], self.multiHeadAttention.n_head, -1).permute(
                0, 2, 1, 3
            )
            * scale
        )
        k = (
            k.view(k.shape[0], k.shape[1], self.multiHeadAttention.n_head, -1).permute(
                0, 2, 3, 1
            )
            * scale
        )
        v = v.view(v.shape[0], v.shape[1], self.multiHeadAttention.n_head, -1).permute(
            0, 2, 1, 3
        )

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        return self.multiHeadAttention.out(wv)


class MultiHeadAttentionSelf(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,  # (b, n_ctx      , n_state)
        k_cache: Tensor,  # (b, n_ctx_cache, n_state)
        v_cache: Tensor,  # (b, n_ctx_cache, n_state)
        mask: Tensor,
    ):
        q = self.multiHeadAttention.query(x)  # (b, n_ctx, n_state)
        k = self.multiHeadAttention.key(x)  # (b, n_ctx, n_state)
        v = self.multiHeadAttention.value(x)  # (b, n_ctx, n_state)

        k_cache[:, -k.shape[1] :, :] = k  # (b, n_ctx_cache + n_ctx, n_state)
        v_cache[:, -v.shape[1] :, :] = v  # (b, n_ctx_cache + n_ctx, n_state)

        k = k_cache
        v = v_cache

        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.multiHeadAttention.n_head) ** -0.25
        q = (
            q.view(q.shape[0], q.shape[1], self.multiHeadAttention.n_head, -1).permute(
                0, 2, 1, 3
            )
            * scale
        )
        k = (
            k_cache.view(
                k.shape[0], k.shape[1], self.multiHeadAttention.n_head, -1
            ).permute(0, 2, 3, 1)
            * scale
        )
        v = v.view(v.shape[0], v.shape[1], self.multiHeadAttention.n_head, -1).permute(
            0, 2, 1, 3
        )

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        return self.multiHeadAttention.out(wv), k_cache, v_cache


class ResidualAttentionBlockTensorCache(nn.Module):
    def __init__(self, inResidualAttentionBlock: ResidualAttentionBlock):
        super().__init__()
        self.originalBlock = inResidualAttentionBlock
        self.attn = MultiHeadAttentionSelf(inResidualAttentionBlock.attn)
        self.cross_attn = (
            MultiHeadAttentionCross(inResidualAttentionBlock.cross_attn)
            if inResidualAttentionBlock.cross_attn
            else None
        )

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        mask: Tensor,
    ):
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(
            self.originalBlock.attn_ln(x), self_k_cache, self_v_cache, mask=mask
        )
        x = x + self_attn_x

        if self.cross_attn is not None:
            x = x + self.cross_attn(
                self.originalBlock.cross_attn_ln(x), cross_k, cross_v
            )

        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x))
        return x, self_k_cache_updated, self_v_cache_updated


class TextDecoderTensorCache(nn.Module):
    def __init__(self, inTextDecoder: TextDecoder, in_n_ctx: int):
        super().__init__()
        self.textDecoder = inTextDecoder
        self.n_ctx = in_n_ctx

        self.blocks = nn.ModuleList()
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlockTensorCache(orginal_block))

    def forward(
        self,
        tokens: Tensor,
        n_layer_self_k_cache: Tensor,
        n_layer_self_v_cache: Tensor,
        n_layer_cross_k: Tensor,
        n_layer_cross_v: Tensor,
        offset: Tensor,
    ):
        offset = offset.int()
        x = (
            self.textDecoder.token_embedding(tokens)
            + self.textDecoder.positional_embedding[
                offset[0] : offset[0] + tokens.shape[-1]
            ]
        )
        x = x.to(n_layer_cross_k[0].dtype)

        i = 0
        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :]
            self_v_cache = n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :]
            x, self_k_cache, self_v_cache = block(
                x,
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=n_layer_cross_k[i],
                cross_v=n_layer_cross_v[i],
                mask=self.textDecoder.mask,
            )
            n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_k_cache
            n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_v_cache
            i += 1

        x = self.textDecoder.ln(x)

        if False:
            # x.shape (1, 3, 384)
            # weight.shape (51684, 384)

            logits = (
                x
                @ torch.transpose(
                    self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1
                )
            ).float()
        else:
            logits = (
                torch.matmul(
                    self.textDecoder.token_embedding.weight.to(x.dtype),
                    x.permute(0, 2, 1),
                )
                .permute(0, 2, 1)
                .float()
            )

        return logits, n_layer_self_k_cache, n_layer_self_v_cache


@torch.jit.export
def MultiHeadAttentionForwardEncoder(self, x: torch.Tensor) -> torch.Tensor:
    q = self.query(x)

    k = self.key(x)
    v = self.value(x)

    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.view(q.shape[0], q.shape[1], self.n_head, -1).permute(0, 2, 1, 3) * scale
    k = k.view(k.shape[0], k.shape[1], self.n_head, -1).permute(0, 2, 3, 1) * scale
    v = v.view(v.shape[0], v.shape[1], self.n_head, -1).permute(0, 2, 1, 3)

    qk = q @ k

    w = F.softmax(qk, dim=-1).to(q.dtype)
    wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

    return self.out(wv)


@torch.jit.export
def ResidualAttentionBlockForwardEncoder(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn.forward_encoder(self.attn_ln(x))[0]
    x = x + self.mlp(self.mlp_ln(x))
    return x


def AudioEncoderForward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
    x = (x + self.positional_embedding).to(x.dtype)

    for block in self.blocks:
        x = block.forward_encoder(x)

    x = self.ln_post(x)
    return x


class Whisper(torch.nn.Module):
    def __init__(self, whisper):
        super().__init__()
        self.encoder = AudioEncoderTensorCache(whisper.encoder, whisper.decoder)
        self.decoder = TextDecoderTensorCache(whisper.decoder, whisper.dims.n_text_ctx)

    @torch.jit.ignore()
    def forward():
        pass

    @torch.jit.export
    def run_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: A 3-D torch tensor of shape (batch_size, dim, T)
        Returns:
          Return a tuple of two tensors:
            - n_layer_cross_k: A 4-D tensor of shape (num_layers, batch_size, T, dim)
            - n_layer_cross_v: A 4-D tensor of shape (num_layers, batch_size, T, dim)
        """
        return self.encoder(x)

    @torch.jit.export
    def run_decoder(
        self,
        tokens: torch.Tensor,
        n_layer_self_k_cache: torch.Tensor,
        n_layer_self_v_cache: torch.Tensor,
        n_layer_cross_k: torch.Tensor,
        n_layer_cross_v: torch.Tensor,
        offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          tokens: A 2-D tensor of shape (batch_size, num_tokens)
          n_layer_self_k_cache: A 4-D tensor of shape (num_layers, batch_size, T, dim)
          n_layer_self_v_cache: A 4-D tensor of shape (num_layers, batch_size, T, dim)
          n_layer_cross_k: A 4-D tensor of shape (num_layers, batch_size, T, dim)
          n_layer_cross_v: A 4-D tensor of shape (num_layers, batch_size, T, dim)
          offset: A 1-D tensor of shape (batch_size,)
        Returns:
          Return a tuple of 3 tensors:
            - logits: A 3-D tensor of shape (batch_size, num_tokens, dim)
            - next_n_layer_self_k_cache, same shape as n_layer_self_k_cache
            - next_n_layer_self_v_cache, same shape as n_layer_self_v_cache
        """
        return self.decoder(
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        )


def main():
    args = get_args()
    name = args.model
    print(args)
    print(name)
    model = whisper.load_model(name)
    n = 0
    for k, v in model.state_dict().items():
        print(k)
        n += v.numel()
    print("n", n, n / 1024 / 1024)

    model.decoder.blocks[0].attn.__class__.forward = torch.jit.ignore(
        model.decoder.blocks[0].attn.__class__.forward
    )

    model.decoder.blocks[0].cross_attn.__class__.forward = torch.jit.ignore(
        model.decoder.blocks[0].cross_attn.__class__.forward
    )

    model.encoder.blocks[0].attn.__class__.forward = torch.jit.ignore(
        model.encoder.blocks[0].attn.__class__.forward
    )

    model.encoder.blocks[0].__class__.forward = torch.jit.ignore(
        model.encoder.blocks[0].__class__.forward
    )

    #  model.encoder.__class__.forward = torch.jit.ignore(model.encoder.__class__.forward)
    model.decoder.__class__.forward = torch.jit.ignore(model.decoder.__class__.forward)

    d = {}
    for name, m in model.named_modules():
        if isinstance(m, LayerNorm):
            d[name] = ModifiedLayerNorm(m)
        elif isinstance(m, Conv1d):
            d[name] = ModifiedConv1d(m)

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(get_submodule(model, parent), child, v)
        else:
            setattr(model, k, v)

    w = Whisper(model)

    m = torch.jit.script(w)
    m.save("model.pt")

    num_param = sum([p.numel() for p in w.parameters()])
    print(f"Number of model parameters: {num_param}")


if __name__ == "__main__":
    ResidualAttentionBlock.forward_encoder = ResidualAttentionBlockForwardEncoder
    MultiHeadAttention.forward_encoder = MultiHeadAttentionForwardEncoder
    AudioEncoder.forward = AudioEncoderForward
    main()
