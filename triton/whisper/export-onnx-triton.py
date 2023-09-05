#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
# Copyright    2023  Nvidia.        (authors: Yuekai Zhang)
# flake8: noqa

"""
Note: Code in this file is modified from
https://github.com/TadaoYamaoka/whisper/blob/main/to_onnx.py

Thanks to https://github.com/TadaoYamaoka
for making the onnx export script public.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

import whisper
from whisper.model import (
    AudioEncoder,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
)

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
            "large-v2"],
        # fmt: on
    )
    return parser.parse_args()

class AudioEncoderTensorCache(nn.Module):
    def __init__(self, inAudioEncoder: AudioEncoder, inTextDecoder: TextDecoder):
        super().__init__()
        self.audioEncoder = inAudioEncoder
        self.textDecoder = inTextDecoder

    def forward(self, x: Tensor):
        audio_features = self.audioEncoder(x)

        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        for block in self.textDecoder.blocks:
            n_layer_cross_k_list.append(block.cross_attn.key(audio_features))
            n_layer_cross_v_list.append(block.cross_attn.value(audio_features))

        # transpose the list of tensors, since we would like it to use batch first
        return torch.stack(n_layer_cross_k_list).transpose(0, 1), torch.stack(n_layer_cross_v_list).transpose(0, 1)


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
        wv, qk = self.multiHeadAttention.qkv_attention(q, k, v, mask)
        return self.multiHeadAttention.out(wv)


class MultiHeadAttentionSelf(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention
        self.n_head = self.multiHeadAttention.n_head

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

        # wv, qk = self.multiHeadAttention.qkv_attention(q, k_cache, v_cache, mask)
        wv, qk = self.qkv_attention(q, k_cache, v_cache, mask)
        return self.multiHeadAttention.out(wv), k_cache, v_cache

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            mask = torch.triu(torch.ones(qk.shape[0],qk.shape[1], n_ctx, n_ctx), diagonal=1).bool().to(qk.device)
            qk = qk.masked_fill(mask, -np.inf)
            # add is not conformable with the broadcasting rules in TensorRT 9.0, 8.6, 8.5
            # qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


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

        if self.cross_attn:
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

        self.blocks = []
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
        # shape of offset tensor is (B, 1), since trtion onnx can't accept scalar tensor
        offset = offset[0]

        x = (
            self.textDecoder.token_embedding(tokens)
            + self.textDecoder.positional_embedding[
                offset[0] : offset[0] + tokens.shape[-1]
            ]
        )
        x = x.to(n_layer_cross_k[0].dtype)

        i = 0
        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[:, i, : offset[0] + tokens.shape[-1], :]
            self_v_cache = n_layer_self_v_cache[:, i, : offset[0] + tokens.shape[-1], :]
            x, self_k_cache, self_v_cache = block(
                x,
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=n_layer_cross_k[:,i],
                cross_v=n_layer_cross_v[:,i],
                mask=self.textDecoder.mask,
            )
            n_layer_self_k_cache[:, i, : offset[0] + tokens.shape[-1], :] = self_k_cache
            n_layer_self_v_cache[:, i, : offset[0] + tokens.shape[-1], :] = self_v_cache
            i += 1

        x = self.textDecoder.ln(x)

        logits = (
            x
            @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits, n_layer_self_k_cache, n_layer_self_v_cache

@torch.no_grad()
def main():
    args = get_args()
    name = args.model

    opset_version = 14

    model = whisper.load_model(name)

    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    model.eval()

    audio = torch.rand(16000 * 2)
    audio = whisper.pad_or_trim(audio)
    assert audio.shape == (16000 * 30,), audio.shape

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device).unsqueeze(0)
    batch_size = 1
    assert mel.shape == (batch_size, 80, 30 * 100)
    n_audio_ctx = mel.shape[-1] // 2
    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)
    n_layer_cross_k, n_layer_cross_v = encoder(mel)
    assert n_layer_cross_k.shape == (
        batch_size,
        model.dims.n_text_layer,
        n_audio_ctx,
        model.dims.n_text_state,
    ), n_layer_cross_k.shape
    assert n_layer_cross_v.shape == (
        batch_size,
        model.dims.n_text_layer,
        n_audio_ctx,
        model.dims.n_text_state,
    ), n_layer_cross_v.shape

    encoder_filename = f"{name}-encoder.onnx"
    torch.onnx.export(
        encoder,
        mel,
        encoder_filename,
        opset_version=opset_version,
        input_names=["mel"],
        output_names=["n_layer_cross_k", "n_layer_cross_v"],
        dynamic_axes={
            "mel": {0: "n_audio", 1: "audio_len"},  # n_audio is also known as batch_size
            "n_layer_cross_k": {0: "n_audio", 2: "n_audio_ctx"},
            "n_layer_cross_v": {0: "n_audio", 2: "n_audio_ctx"},
        },
    )

    n_audio = mel.shape[0]
    tokens = torch.tensor([[tokenizer.sot, tokenizer.sot, tokenizer.sot]] * n_audio).to(
        mel.device
    )  # [n_audio, 3]
    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)

    # make the tensors be batch first
    n_layer_self_k_cache = torch.zeros(
        (   
            n_audio,
            len(model.decoder.blocks),
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    n_layer_self_v_cache = torch.zeros(
        (
            n_audio,
            len(model.decoder.blocks),
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )

    offset = torch.tensor([[0]], dtype=torch.int64).to(mel.device) # [n_audio, 1]
    decoder_filename = f"{name}-decoder.onnx"
    torch.onnx.export(
        decoder,
        (
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        ),
        decoder_filename,
        opset_version=opset_version,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
        ],
        output_names=["logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"],
        dynamic_axes={
            "tokens": {0: "n_audio", 1: "n_tokens"},
            "in_n_layer_self_k_cache": {0: "n_audio", 2: "n_text_ctx"},
            "in_n_layer_self_v_cache": {0: "n_audio", 2: "n_text_ctx"},
            "n_layer_cross_k": {0: "n_audio", 2: "n_audio_ctx"},
            "n_layer_cross_v": {0: "n_audio", 2: "n_audio_ctx"},
            "offset": {0: "n_audio"},
        },
    )

if __name__ == "__main__":
    main()
