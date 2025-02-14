# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack
import torchaudio
import jieba
import triton_python_backend_utils as pb_utils
from pypinyin import Style, lazy_pinyin
import math
import os
from functools import wraps
from typing import List
import tensorrt as trt

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.session import Session


# import onnxruntime
import numpy as np


def get_vocos_mel_spectrogram(
    waveform, # B,T
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel

def get_tokenizer(vocab_file_path: str):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size

def convert_char_to_pinyin(reference_target_texts_list, polyphone=True):
    final_reference_target_texts_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"  # common chinese characters

    for text in reference_target_texts_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(
                seg
            ):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(
                            lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                        )
                    else:
                        char_list.append(c)
        final_reference_target_texts_list.append(char_list)

    return final_reference_target_texts_list

def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
):  # noqa: F722
    list_idx_tensors = [
        torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text
    ]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers = 0, conv_mult = 2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.precompute_max_pos = 4096  # ~44s of 24khz audio
        self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
        self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])

    def forward(self, text, seq_len):
        text = F.pad(text, (0, seq_len - text.shape[1]), mode='constant')
        text = self.text_embed(text) # b n -> b n d
        text = self.text_blocks(text + self.freqs_cis[:text.shape[1], :])
        return text

def lens_to_mask(
    t, length: int | None = None  # noqa: F722 F821
):  # noqa: F722 F821
    if not length:
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

class F5TTS(object):

    def __init__(self,
                 config,
                 debug_mode=True,
                 stream: torch.cuda.Stream = None,
                 tllm_model_dir: str = None,
                 ):
        self.dtype = config['pretrained_config']['dtype']

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(world_size=world_size,
                                            rank=rank,
                                            cp_size=cp_size,
                                            tp_size=tp_size,
                                            pp_size=1,
                                            gpus_per_node=1)

        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f"cuda:{local_rank}")

        torch.cuda.set_device(self.device)

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_file = os.path.join(tllm_model_dir, f"rank{rank}.engine")
        logger.info(f'Loading engine from {engine_file}')
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()

        assert engine_buffer is not None

        self.session = Session.from_serialized_engine(engine_buffer)

        self.debug_mode = debug_mode

        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        expected_tensor_names = ['noise', 'cond', 'cond_drop', 'time', 'rope_cos', 'rope_sin', 't_scale', 'denoised']

        if self.mapping.tp_size > 1:
            self.buffer, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.mapping.tp_size))
            self.inputs['all_reduce_workspace'] = self.all_reduce_workspace
            expected_tensor_names += ['all_reduce_workspace']

        found_tensor_names = [
            self.session.engine.get_tensor_name(i)
            for i in range(self.session.engine.num_io_tensors)
        ]
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected.")
        if self.debug_mode:
            self.debug_tensors = list(
                set(found_tensor_names) - set(expected_tensor_names))

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _setup(self, batch_size):
        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.session.engine.get_tensor_mode(
                    name) == trt.TensorIOMode.OUTPUT:
                shape = list(self.session.engine.get_tensor_shape(name))
                shape[1] = batch_size
                self.outputs[name] = torch.empty(shape,
                                                 dtype=self._tensor_dtype(name),
                                                 device=self.device)

        self.buffer_allocated = True

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    @cuda_stream_guard
    def forward(self, noise: torch.Tensor, cond: torch.Tensor,
                cond_drop: torch.Tensor, time_expand: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor, delta_t: torch.Tensor):

        self._setup(noise.shape[1])
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        time = time_expand[:, 0]
        t_scale = delta_t[0].unsqueeze(0)
        input_type = str_dtype_to_torch(self.dtype)
        inputs = {
            'noise': noise.to(input_type),
            'cond': cond.to(input_type),
            'cond_drop': cond_drop.to(input_type),
            'time': time.to(input_type),
            'rope_cos': rope_cos.to(input_type),
            'rope_sin': rope_sin.to(input_type),
            't_scale': t_scale.to(input_type)
        }
        self.inputs.update(**inputs)
        self.session.set_shapes(self.inputs)

        for tensor_name in self.inputs:
            tensor = self.inputs[tensor_name]
            ptr = tensor.data_ptr()
            self.session.context.set_tensor_address(tensor_name, ptr)

        for tensor_name in self.outputs:
            tensor = self.outputs[tensor_name]
            ptr = tensor.data_ptr() if isinstance(tensor,
                                                torch.Tensor) else tensor
            self.session.context.set_tensor_address(tensor_name, ptr)

        i = 0
        while i < 16:
            if 0 != i:
                self.inputs['time'] = time_expand[:, i].to(input_type)
                self.inputs['t_scale'] = delta_t[i].unsqueeze(0).to(input_type)
                self.inputs['noise'] = self.outputs["denoised"]
                self.session.context.set_tensor_address('time', self.inputs['time'].data_ptr())
                self.session.context.set_tensor_address('t_scale', self.inputs['t_scale'].data_ptr())
                self.session.context.set_tensor_address('noise', self.inputs['noise'].data_ptr())
            self.session.context.execute_async_v3(self.stream.cuda_stream)
            i += 1
        return self.outputs["denoised"]

def load_checkpoint(ckpt_path, use_ema=True):
    checkpoint = torch.load(ckpt_path, weights_only=True)
    if use_ema:
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

    dict = checkpoint["model_state_dict"]
    text_embed_dict = {}
    for key in dict.keys():
        # transformer.text_embed.text_embed.weight -> text_embed.weight
        if 'text_embed' in key:
            text_embed_dict[key.replace('transformer.text_embed.', '')] = dict[key]
    return text_embed_dict

class TritonPythonModel:
    def initialize(self, args):

        self.reference_sample_rate = 16000

        parameters = json.loads(args['model_config'])['parameters']
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        self.vocab_char_map, self.vocab_size = get_tokenizer(parameters["vocab_file"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_embedding = TextEmbedding(text_num_embeds=self.vocab_size, text_dim=512, conv_layers=4).to(self.device)
        # load weights from f5_model.transformer.text_embedding
        self.text_embedding.load_state_dict(load_checkpoint(parameters["model_path"]), strict=True)

        self.target_rms = 0.15 # rms means root mean square

        self.base_rescale_factor = 1.0
        self.interpolation_factor = 1.0
        self.head_dim = 64
        MAX_SIGNAL_LENGTH = 2048 
        base = 10000.0 * self.base_rescale_factor ** (self.head_dim / (self.head_dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(MAX_SIGNAL_LENGTH, dtype=torch.float32), inv_freq) / self.interpolation_factor
        self.freqs = freqs.repeat_interleave(2, dim=-1).unsqueeze(0)
        self.rope_cos = self.freqs.cos().half()
        self.rope_sin = self.freqs.sin().half()

        # self.resampler = torchaudio.transforms.Resample(16000, 24000)
        self.resampler = torchaudio.transforms.Resample(self.reference_sample_rate, 24000)

        self.tllm_model_dir = parameters["tllm_model_dir"]

        tensorrt_llm.logger.set_level("info")
        torch.manual_seed(0)

        # Load model:
        config_file = os.path.join(self.tllm_model_dir, 'config.json')
        with open(config_file) as f:
            config = json.load(f)

        self.model = F5TTS(config, debug_mode=False, tllm_model_dir=self.tllm_model_dir)

    def forward_vocoder(self, mel):
        input_tensor_0 = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel))
    
        inference_request = pb_utils.InferenceRequest(
            model_name='vocoder',
            requested_output_names=['waveform'],
            inputs=[input_tensor_0])
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            waveform = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'waveform')
            waveform = torch.utils.dlpack.from_dlpack(waveform.to_dlpack()).cpu()
            return waveform

    def execute(self, requests):
        reference_target_texts_list, reference_wavs_tensor, estimated_reference_target_mel_len, reference_mel_len = [], [], [], []
        max_wav_len = 0
        for request in requests:
            wav_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            wav_lens = pb_utils.get_input_tensor_by_name(
                request, "reference_wav_len")
            reference_text = pb_utils.get_input_tensor_by_name(
                request, "reference_text").as_numpy()
            reference_text = reference_text[0][0].decode('utf-8')
            target_text = pb_utils.get_input_tensor_by_name(
                request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')

            text = reference_text + target_text
            reference_target_texts_list.append(text)
         
            # Move WAV data to GPU
            wav = from_dlpack(wav_tensor.to_dlpack())
            assert wav.shape[0] == 1, "Only support batch size 1 for now."
            wav_len = from_dlpack(wav_lens.to_dlpack())
            wav_len = wav_len.squeeze()
            # wav_len  = int(wav_len / 16000 * 24000)
            wav_len  = int(wav_len / self.reference_sample_rate * 24000)

            wav = self.resampler(wav)

            wav = wav[:, :wav_len]

            ref_rms = torch.sqrt(torch.mean(torch.square(wav)))

            wav = wav * self.target_rms / ref_rms

                

            max_wav_len = max(max_wav_len, wav_len)
            reference_wavs_tensor.append(wav)
            
            estimated_reference_target_mel_len.append(int(wav_len // 256) + int(wav_len // 256 * len(target_text) / len(reference_text)))
            reference_mel_len.append(int(wav_len // 256))
        
        max_seq_len = max(estimated_reference_target_mel_len)
        # max_seq_len = 884
        reference_wavs_tensor = torch.cat(
            [F.pad(wav, (0, max_wav_len - wav.shape[1]), mode='constant') for wav in reference_wavs_tensor]
        ).to(self.device)
        print(f"reference_wavs_tensor shape: {reference_wavs_tensor.shape}")
        print(reference_wavs_tensor[0,:100])
        mel_features = get_vocos_mel_spectrogram(reference_wavs_tensor)
        mel_features = mel_features.transpose(1, 2)
        # pad to max_seq_len
        mel_features = F.pad(mel_features, (0, 0, 0, max_seq_len - mel_features.shape[1]), mode='constant')

        pinyin_list = convert_char_to_pinyin(reference_target_texts_list, polyphone=True)
        text_pad_sequence = list_str_to_idx(pinyin_list, self.vocab_char_map).to(self.device)
        text_pad_sequence += 1

        # text_pad_sequence B,T,D, now text_pad_sequence_drop B+1, T,D append torch.zeros(1, T, D) at the bottom
        text_pad_sequence_drop = torch.cat((text_pad_sequence, torch.zeros((1, text_pad_sequence.shape[1]), dtype=torch.int32).to(self.device)), dim=0)
        # text_embedding = self.text_embedding(text_pad_sequence, max_seq_len)
        text_embedding_drop_condition = self.text_embedding(text_pad_sequence_drop, max_seq_len)
        text_embedding = text_embedding_drop_condition[:-1]
        text_embedding_drop = text_embedding_drop_condition[-1].unsqueeze(0)


        # mask = lens_to_mask(torch.tensor(estimated_reference_target_mel_len))

        noise = torch.randn_like(mel_features).to(self.device)
        rope_cos = self.rope_cos[:, :max_seq_len, :].float()
        rope_sin = self.rope_sin[:, :max_seq_len, :].float()

        cat_mel_text = torch.cat((mel_features, text_embedding), dim=-1)


        batch = cat_mel_text.shape[0]
        assert mel_features.shape[0] == 1, "Only support batch size 1 for now."
        # cat_mel_text_drop = torch.cat((torch.zeros((batch, max_seq_len, 100), dtype=torch.float32).to(self.device), self.text_embedding(torch.zeros((batch, max_seq_len), dtype=torch.int32).to(self.device), max_seq_len)), dim=-1)
        cat_mel_text_drop = torch.cat((torch.zeros((batch, max_seq_len, 100), dtype=torch.float32).to(self.device), text_embedding_drop), dim=-1)
        t = torch.linspace(0, 1, 16 + 1, dtype=torch.float32)
        time_step = t + (-1.0) * (torch.cos(torch.pi * 0.5 * t) - 1 + t)

        delta_t = torch.diff(time_step)
        
        time_expand = torch.zeros((batch, 16, 256), dtype=torch.float32)
        half_dim = 256 // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
        for i in range(16):
            emb = time_step[i] * emb_factor
            time_expand[:, i, :] = torch.cat((emb.sin(), emb.cos()), dim=-1)

        denoised = self.model.forward(noise.cuda(),cat_mel_text.cuda(), cat_mel_text_drop.cuda(), time_expand.cuda(), rope_cos.cuda(), rope_sin.cuda(), delta_t.cuda())

        ref_me_len = reference_mel_len[0]
        denoised_bls = denoised[:, ref_me_len:, :].transpose(1, 2).to(torch.float32).contiguous().cpu()

        audios = self.forward_vocoder(denoised_bls)
        rms = torch.sqrt(torch.mean(torch.square(audios)))
        audios = audios * self.target_rms / rms

        responses = []
        for i, item in enumerate(audios):
            waveform = item.unsqueeze(0)
            print(f"waveform shape: {waveform.shape}")
            waveform = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(waveform))
            inference_response = pb_utils.InferenceResponse(output_tensors=[waveform])
            responses.append(inference_response)
                             
        return responses
