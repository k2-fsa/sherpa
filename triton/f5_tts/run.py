# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#               2025               （authors: Yuekai Zhang）
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
# Modified from https://github.com/xingchensong/S3Tokenizer/blob/main/s3tokenizer/cli.py
""" Example Usage
split=test_zh
llm_path=f5-tts/exp_zh/checkpoint-805000
huggingface-cli download --local-dir f5-tts-small-wenetspeech4tts-basic yuekai/f5-tts-semantic-token-small-wenetspeech4tts-basic
model_path=f5-tts-small-wenetspeech4tts-basic/epoch-10-avg-5.pt
huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x --local-dir ./bigvgan_v2_24khz_100band_256x
vocoder=./bigvgan_v2_24khz_100band_256x
torchrun --nproc_per_node=2 \
    f5-tts/infer_dist.py \
                --output_dir $output_dir \
                --batch_size 1 \
                --num_workers 2 \
                --llm-model-name-or-path $llm_path \
                --flow-matching-model-path $model_path \
                --decoder-dim 768 --nhead 12 --num-decoder-layers 18 \
                --use-cosyvoice-semantic-token True \
                --vocoder-dir $vocoder \
                --split-name $split -top-k 50 -top-p 0.95 -temperature 0.8 \
                --tokenizer-dir Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
import math
import time
from pathlib import Path
from functools import wraps
from typing import List, Dict, Union, Optional

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import jieba
from pypinyin import Style, lazy_pinyin
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from vocos import Vocos
import tensorrt as trt

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.session import Session

torch.manual_seed(0)
def get_args():
    parser = argparse.ArgumentParser(description="extract speech code")
    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        choices=["wenetspeech4tts", "test_zh", "test_en", "test_hard"],
        help="huggingface dataset split name",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="dir to save result"
    )
    parser.add_argument(
        "--vocab-file",
        required=True,
        type=str,
        help="vocab file",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--tllm-model-dir",
        required=True,
        type=str,
        help="tllm model dir",
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        help="batch size (per-device) for inference",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="workers for dataloader"
    )
    parser.add_argument(
        "--prefetch", type=int, default=None, help="prefetch for dataloader"
    )
    parser.add_argument(
        "--vocoder",
        default="vocos",
        type=str,
        help="vocoder name",
    )
    parser.add_argument('--enable-warmup', action='store_true')
    parser.add_argument('--remove-input-padding', action='store_true')
    parser.add_argument('--use-perf', action='store_true')
    # add_model_arguments(parser)
    args = parser.parse_args()
    return args


def padded_mel_batch(ref_mels, max_seq_len):
    padded_ref_mels = []
    for mel in ref_mels:
        # pad along the last dimension
        padded_ref_mel = F.pad(mel, (0, 0, 0, max_seq_len - mel.shape[0]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    return padded_ref_mels


def data_collator(batch, vocab_char_map, device="cuda"):
    use_perf = True
    if use_perf:
        torch.cuda.nvtx.range_push("data_collator")
    target_sample_rate = 24000
    hop_length = 256
    target_rms = 0.1
    ids, ref_mel_list, ref_mel_len_list, estimated_reference_target_mel_len, reference_target_texts_list = [], [], [], [], []
    for i, item in enumerate(batch):
        item_id, prompt_text, target_text = (
            item["id"],
            item["prompt_text"],
            item["target_text"],
        )
        ids.append(item_id)
        reference_target_texts_list.append(prompt_text + target_text)

        ref_audio_org, ref_sr = (
            item["prompt_audio"]["array"],
            item["prompt_audio"]["sampling_rate"],
        )
        ref_audio_org = torch.from_numpy(ref_audio_org).unsqueeze(0).float()
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio_org)))
        if ref_rms < target_rms:
            ref_audio_org = ref_audio_org * target_rms / ref_rms

        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio_org)
        else:
            ref_audio = ref_audio_org

        # to mel spectrogram
        if use_perf:
            torch.cuda.nvtx.range_push(f"mel_spectrogram {i}")
        ref_mel = mel_spectrogram(ref_audio, vocoder="vocos", device="cuda")
        if use_perf:
            torch.cuda.nvtx.range_pop()
        ref_mel = ref_mel.squeeze()
        ref_mel_len = ref_mel.shape[0]
        assert ref_mel.shape[1] == 100

        ref_mel_list.append(ref_mel)
        ref_mel_len_list.append(ref_mel_len)

        estimated_reference_target_mel_len.append(int(ref_mel.shape[0] * (1 + len(target_text) / len(prompt_text))))

    max_seq_len = max(estimated_reference_target_mel_len)
    ref_mel_batch = padded_mel_batch(ref_mel_list, max_seq_len)
    ref_mel_len_batch = torch.LongTensor(ref_mel_len_list)

    pinyin_list = convert_char_to_pinyin(reference_target_texts_list, polyphone=True)
    text_pad_sequence = list_str_to_idx(pinyin_list, vocab_char_map)
    
    for i, item in enumerate(text_pad_sequence):
        text_pad_sequence[i] = F.pad(item, (0, estimated_reference_target_mel_len[i] - len(item)), mode='constant', value=-1)
        text_pad_sequence[i] += 1  # WAR: 0 is reserved for padding token, hard coding in F5-TTS
    text_pad_sequence = pad_sequence(text_pad_sequence, padding_value=-1, batch_first=True).to(device)
    text_pad_sequence = F.pad(text_pad_sequence, (0, max_seq_len - text_pad_sequence.shape[1]), mode='constant', value=-1)
    if use_perf:
        torch.cuda.nvtx.range_pop()
    return {
        "ids": ids,
        "ref_mel_batch": ref_mel_batch,
        "ref_mel_len_batch": ref_mel_len_batch,
        "text_pad_sequence": text_pad_sequence,
        "estimated_reference_target_mel_len": estimated_reference_target_mel_len,
    }


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    print(
        "Inference on multiple gpus, this gpu {}".format(local_rank)
        + ", rank {}, world_size {}".format(rank, world_size)
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


class F5TTS(object):

    def __init__(
        self,
        config,
        debug_mode=True,
        stream: Optional[torch.cuda.Stream] = None,
        tllm_model_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
    ):
        self.dtype = config['pretrained_config']['dtype']

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(
            world_size=world_size,
            rank=rank,
            cp_size=cp_size,
            tp_size=tp_size,
            pp_size=1,
            gpus_per_node=1
        )

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

        expected_tensor_names = ['noise', 'cond', 'time', 'rope_cos', 'rope_sin', 'input_lengths', 'denoised']

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
  
        self.max_mel_len = 4096
        self.text_embedding = TextEmbedding(
            text_num_embeds=vocab_size, 
            text_dim=512, 
            conv_layers=4, 
            precompute_max_pos=self.max_mel_len
        ).to(self.device)
        self.text_embedding.load_state_dict(load_checkpoint(model_path), strict=True)

        self.target_audio_sample_rate = 24000
        self.target_rms = 0.15  # target rms for audio
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.n_mel_channels = 100
        #self.max_mel_len = 3000
        self.head_dim = 64
        self.base_rescale_factor = 1.0
        self.interpolation_factor = 1.0
        base = 10000.0 * self.base_rescale_factor ** (self.head_dim / (self.head_dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(self.max_mel_len, dtype=torch.float32), inv_freq) / self.interpolation_factor
        self.freqs = freqs.repeat_interleave(2, dim=-1).unsqueeze(0)
        self.rope_cos = self.freqs.cos().half()
        self.rope_sin = self.freqs.sin().half()
        self.nfe_steps = 16

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _setup(self, batch_size, seq_len):
        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.session.engine.get_tensor_mode(
                    name) == trt.TensorIOMode.OUTPUT:
                shape = list(self.session.engine.get_tensor_shape(name))
                shape[0] = batch_size
                shape[1] = seq_len
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
                time_expand: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor, 
                input_lengths: torch.Tensor, delta_t: torch.Tensor, use_perf: bool = False):
        if use_perf:
            torch.cuda.nvtx.range_push("flow matching")
        cfg_strength = 2.0
        batch_size = noise.shape[0]
        half_batch = batch_size // 2
        noise_half = noise[:half_batch]  # Store the initial half of noise
        
        input_type = str_dtype_to_torch(self.dtype)
        
        # Keep a copy of the initial tensors
        cond = cond.to(input_type)
        rope_cos = rope_cos.to(input_type)  
        rope_sin = rope_sin.to(input_type)
        input_lengths = input_lengths.to(str_dtype_to_torch("int32"))
        
        # Instead of iteratively updating noise within a single model context,
        # we'll do a single forward pass for each iteration with fresh context setup
        for i in range(self.nfe_steps):
            # Synchronize before setting up for this iteration
            # torch.cuda.synchronize()
            
            # Re-setup the buffers for clean execution
            self._setup(batch_size, noise.shape[1])
            if not self.buffer_allocated:
                raise RuntimeError('Buffer not allocated, please call setup first!')
                
            # Re-create combined noises for this iteration
            current_noise = torch.cat([noise_half, noise_half], dim=0).to(input_type)
            
            # Get time step for this iteration
            current_time = time_expand[:, i].to(input_type)
            
            # Create fresh input dictionary for this iteration
            current_inputs = {
                'noise': current_noise,
                'cond': cond,
                'time': current_time,
                'rope_cos': rope_cos,
                'rope_sin': rope_sin,
                'input_lengths': input_lengths,
            }
            
            # Update inputs and set shapes
            self.inputs.clear()  # Clear previous inputs
            self.inputs.update(**current_inputs)
            self.session.set_shapes(self.inputs)
            
            # Set tensor addresses
            for tensor_name in self.inputs:
                tensor = self.inputs[tensor_name]
                ptr = tensor.data_ptr()
                self.session.context.set_tensor_address(tensor_name, ptr)
            
            for tensor_name in self.outputs:
                tensor = self.outputs[tensor_name]
                ptr = tensor.data_ptr() if isinstance(tensor, torch.Tensor) else tensor
                self.session.context.set_tensor_address(tensor_name, ptr)
            
            # Execute the model
            if use_perf:
                torch.cuda.nvtx.range_push(f"execute {i}")
            self.session.context.execute_async_v3(self.stream.cuda_stream)
            if use_perf:
                torch.cuda.nvtx.range_pop()
            # Synchronize to ensure completion
            # torch.cuda.synchronize()
            
            # Process results
            t_scale = delta_t[i].unsqueeze(0).to(input_type)
            
            # Extract predictions
            pred_cond = self.outputs["denoised"][:half_batch]
            pred_uncond = self.outputs["denoised"][half_batch:]
            
            # Apply classifier-free guidance with safeguards
            guidance = pred_cond + (pred_cond - pred_uncond) * cfg_strength
            # Calculate update for noise
            noise_half = noise_half + guidance * t_scale
        if use_perf:
            torch.cuda.nvtx.range_pop()
        return noise_half

    def sample(self, text_pad_sequence: torch.Tensor, ref_mel_batch: torch.Tensor, 
               ref_mel_len_batch: torch.Tensor, estimated_reference_target_mel_len: List[int], remove_input_padding: bool = False, use_perf: bool = False):
        if use_perf:
            torch.cuda.nvtx.range_push("text embedding")
        batch = text_pad_sequence.shape[0]
        max_seq_len = ref_mel_batch.shape[1]
        # max_seq_len,"max_seq_len")
        text_pad_sequence_drop = torch.cat(
            (text_pad_sequence, torch.zeros((1, text_pad_sequence.shape[1]), dtype=torch.int32).to(self.device)), 
            dim=0
        )
        
        # text_embedding_drop_condition = self.text_embedding(text_pad_sequence_drop.to(self.device))
        text_embedding_drop_list = []
        for i in range(batch+1):
            text_embedding_drop_list.append(self.text_embedding(text_pad_sequence_drop[i].unsqueeze(0).to(self.device)))
        text_embedding_drop_condition = torch.cat(text_embedding_drop_list, dim=0)
        
        text_embedding = text_embedding_drop_condition[:-1]
        # text_embedding_drop B,T,C batch should be the same
        text_embedding_drop = text_embedding_drop_condition[-1].unsqueeze(0).repeat(batch, 1, 1)

        noise = torch.randn_like(ref_mel_batch).to(self.device)
        rope_cos = self.rope_cos[:, :max_seq_len, :].float().repeat(batch, 1, 1) 
        rope_sin = self.rope_sin[:, :max_seq_len, :].float().repeat(batch, 1, 1)

        cat_mel_text = torch.cat((ref_mel_batch, text_embedding), dim=-1)
        cat_mel_text_drop = torch.cat(
            (torch.zeros((batch, max_seq_len, self.n_mel_channels), dtype=torch.float32).to(self.device), text_embedding_drop), 
            dim=-1
        )
        # below could be reused
        t = torch.linspace(0, 1, self.nfe_steps + 1, dtype=torch.float32)
        time_step = t + (-1.0) * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
        delta_t = torch.diff(time_step)
        # WAR: hard coding 256 here
        tmp_dim = 256
        time_expand = torch.zeros((batch, self.nfe_steps, tmp_dim), dtype=torch.float32)
        half_dim = tmp_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
        for i in range(self.nfe_steps):
            emb = time_step[i] * emb_factor
            time_expand[:, i, :] = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # Convert estimated_reference_target_mel_len to tensor
        input_lengths = torch.tensor(estimated_reference_target_mel_len, dtype=torch.int32)
        
        # combine above along the batch dimension
        inputs = {
            'noise': torch.cat((noise, noise), dim=0).contiguous(),
            'cond': torch.cat((cat_mel_text, cat_mel_text_drop), dim=0).contiguous(),
            'time_expand': torch.cat((time_expand, time_expand), dim=0).contiguous(),
            'rope_cos': torch.cat((rope_cos, rope_cos), dim=0).contiguous(),
            'rope_sin': torch.cat((rope_sin, rope_sin), dim=0).contiguous(),
            'input_lengths': torch.cat((input_lengths, input_lengths), dim=0).contiguous(),
            'delta_t': torch.cat((delta_t, delta_t), dim=0).contiguous()
        }
        if remove_input_padding:
            max_seq_len = inputs['cond'].shape[1]
            inputs['noise'] = remove_tensor_padding(inputs['noise'], inputs['input_lengths'])
            inputs['cond'] = remove_tensor_padding(inputs['cond'], inputs['input_lengths'])
            # for time_expand, convert from B,D to B,T,D by repeat
            inputs['time_expand'] = inputs['time_expand'].unsqueeze(1).repeat(1, max_seq_len, 1, 1)
            inputs['time_expand'] = remove_tensor_padding(inputs['time_expand'], inputs['input_lengths'])
            inputs['rope_cos'] = remove_tensor_padding(inputs['rope_cos'], inputs['input_lengths'])
            inputs['rope_sin'] = remove_tensor_padding(inputs['rope_sin'], inputs['input_lengths'])

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
            print(key, inputs[key].shape)
        if use_perf:
            torch.cuda.nvtx.range_pop()
        start_time = time.time()
        denoised = self.forward(**inputs, use_perf=use_perf)
        cost_time = time.time() - start_time
        print(f"cost time: {cost_time} seconds")
        if remove_input_padding:
            denoised_list = []
            start_idx = 0
            for i in range(batch):
                denoised_list.append(denoised[start_idx:start_idx+inputs['input_lengths'][i]])
                start_idx += inputs['input_lengths'][i]
            return denoised_list, cost_time
        return denoised, cost_time

def remove_tensor_padding(input_tensor,
                          input_tensor_lengths=None):
    # Audio tensor case: batch, seq_len, feature_len
    # position_ids case: batch, seq_len
    assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"

    # Initialize a list to collect valid sequences
    valid_sequences = []

    for i in range(input_tensor.shape[0]):
        valid_length = input_tensor_lengths[i]
        valid_sequences.append(input_tensor[i, :valid_length])

    # Concatenate all valid sequences along the batch dimension
    output_tensor = torch.cat(valid_sequences, dim=0).contiguous()
    return output_tensor

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
    text: Union[List[str], List[List[str]]],
    vocab_char_map: Dict[str, int],  # {char: idx}
    padding_value=-1,
):
    list_idx_tensors = [
        torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text
    ]  # pinyin or char style
    # text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return list_idx_tensors


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


def load_checkpoint(ckpt_path, use_ema=True):
    checkpoint = torch.load(ckpt_path, weights_only=True)
    if use_ema:
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
    dict_state = checkpoint["model_state_dict"]
    text_embed_dict = {}
    for key in dict_state.keys():
        # transformer.text_embed.text_embed.weight -> text_embed.weight
        if 'text_embed' in key:
            text_embed_dict[key.replace('transformer.text_embed.', '')] = dict_state[key]
    return text_embed_dict


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers = 0, conv_mult = 2, precompute_max_pos = 4096):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, precompute_max_pos), persistent=False)
        self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])

    # def forward(self, text):
    #     text_mask = text != -1
    #     # calculate the number of -1 and 0 in text[0]
    #     # num_zero = (text[0] == 0).sum()
    #     # num_minus_one = (text[0] == -1).sum()
    #     # print(num_zero, num_minus_one, "num_zero, num_minus_one")
    #     # print(text_mask.shape, "text_mask", text_mask[0])
    #     # change all -1 to 0
    #     text = text.masked_fill(text == -1, 0)
    #     text = self.text_embed(text)
    #     text = text + self.freqs_cis[:text.shape[1], :]
    #     print(text.shape, "text")
    #     print(text_mask.shape, "text_mask")
    #     text = text.masked_fill(~text_mask.unsqueeze(-1).expand(-1, -1, text.shape[2]), 0)
    #     for block in self.text_blocks:
    #         text = block(text)
    #         text = text.masked_fill(~text_mask.unsqueeze(-1).expand(-1, -1, text.shape[2]), 0)
    #     return text
    def forward(self, text):
        # only keep tensors with value not -1
        text_mask = text != -1
        text_pad_cut_off_index = text_mask.sum(dim=1).max()

        text = text[:, :text_pad_cut_off_index]
        text = self.text_embed(text)
        text = text + self.freqs_cis[:text.shape[1], :]
        for block in self.text_blocks:
            text = block(text)
        # padding text to the original length
        # text shape: B,seq_len,C
        # pad at the second dimension

        text = F.pad(text, (0, 0, 0, text_mask.shape[1] - text.shape[1], 0, 0), value=0)

        return text

def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device="cuda", hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        vocoder = BigVGANInference.from_pretrained(args.vocoder_dir, use_cuda_kernel=False)
    vocoder = vocoder.eval().to(device)
    return vocoder

def mel_spectrogram(waveform, vocoder="vocos", device="cuda"):
    if vocoder == "vocos":
        mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=100,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        ).to(device)
    mel = mel_stft(waveform.to(device))
    mel = mel.clamp(min=1e-5).log()
    return mel.transpose(1, 2)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert torch.cuda.is_available()
    world_size, local_rank, rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")


    vocab_char_map, vocab_size = get_tokenizer(args.vocab_file)

    tllm_model_dir = args.tllm_model_dir
    config_file = os.path.join(tllm_model_dir, 'config.json')
    with open(config_file) as f:
        config = json.load(f)
    model = F5TTS(config, debug_mode=False, tllm_model_dir=tllm_model_dir, model_path=args.model_path, vocab_size=vocab_size)
    
    vocoder = load_vocoder(vocoder_name=args.vocoder, device=device)

    dataset = load_dataset(
        "yuekai/seed_tts",
        split=args.split_name,
        trust_remote_code=True,
    )
    # dataset = dataset.select(range(8))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        collate_fn=lambda x: data_collator(x, vocab_char_map),
    )

    total_steps = len(dataset)

    if args.enable_warmup:
        for batch in dataloader:
            ref_mels, ref_mel_lens = batch["ref_mel_batch"].to(device), batch["ref_mel_len_batch"].to(device)
            text_pad_seq = batch["text_pad_sequence"].to(device)
            total_mel_lens = batch["estimated_reference_target_mel_len"]
            model.sample(text_pad_seq, ref_mels, ref_mel_lens, total_mel_lens, remove_input_padding=args.remove_input_padding)

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    decoding_time = 0
    total_duration = 0
    for batch in dataloader:
        if args.use_perf:
            torch.cuda.cudart().cudaProfilerStart()
        if args.use_perf:
            torch.cuda.nvtx.range_push("data sample")
        ref_mels, ref_mel_lens = batch["ref_mel_batch"].to(device), batch[
            "ref_mel_len_batch"
        ].to(device)
        text_pad_seq = batch["text_pad_sequence"].to(device)
        total_mel_lens = batch["estimated_reference_target_mel_len"]
        # print(text_pad_seq.shape, ref_mels.shape, ref_mel_lens.shape)
        # print(total_mel_lens)
        if args.use_perf:
            torch.cuda.nvtx.range_pop()
        generated, cost_time = model.sample(
            text_pad_seq,
            ref_mels,
            ref_mel_lens,
            total_mel_lens,
            remove_input_padding=args.remove_input_padding,
            use_perf=args.use_perf,
            # steps=16,
            # cfg_strength=2.0,
            # sway_sampling_coef=-1,
            # seed=0,
        )
        decoding_time += cost_time
        for i, gen in enumerate(generated):
            gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
            gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
            # print(gen_mel_spec.shape, gen_mel_spec[0])
            if args.vocoder == "vocos":
                if args.use_perf:
                    torch.cuda.nvtx.range_push(f"vocoder decode")
                generated_wave = vocoder.decode(gen_mel_spec).cpu()
                print(2333333333333333333333333333333333333333333333)
                if args.use_perf:
                    torch.cuda.nvtx.range_pop()
            else:
                generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
            target_rms = 0.1
            target_sample_rate = 24_000
            # if ref_rms_list[i] < target_rms:
            #     generated_wave = generated_wave * ref_rms_list[i] / target_rms
            rms = torch.sqrt(torch.mean(torch.square(generated_wave)))
            if rms < target_rms:
                generated_wave = generated_wave * target_rms / rms
            utt = batch["ids"][i]
            torchaudio.save(
                f"{args.output_dir}/{utt}.wav",
                generated_wave,
                target_sample_rate,
            )
            total_duration += generated_wave.shape[1] / target_sample_rate
        if rank == 0:
            progress_bar.update(world_size * len(batch["ids"]))

    if rank == 0:
        progress_bar.close()

    rtf = decoding_time / total_duration
    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {decoding_time:.3f} seconds " f"({decoding_time/3600:.2f} hours)\n"
    s += f"batch size: {args.batch_size}\n"
    print(s)

    with open(f"{args.output_dir}/rtf.txt", "w") as f:
        f.write(s)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
