
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections import OrderedDict
from pathlib import Path

from typing import List, Dict, Any, Tuple
from .fbank import FeatureExtractor
import torch
import torch.nn as nn
from torch.utils.dlpack import from_dlpack, to_dlpack

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
import triton_python_backend_utils as pb_utils
import math

def remove_tensor_padding(input_tensor,
                          input_tensor_lengths=None,
                          pad_value=None):
    if pad_value:
        assert input_tensor_lengths is None, "input_tensor_lengths should be None when pad_value is provided"
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] != pad_value
        ), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    else:
        # Audio tensor case: batch, seq_len, feature_len
        # position_ids case: batch, seq_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(input_tensor.shape[0]):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)
    return output_tensor

def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config

class WhisperEncoding:

    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config = read_config('encoder', engine_dir)
        self.n_mels = config['n_mels']
        self.dtype = config['dtype']
        self.num_languages = config['num_languages']
        self.encoder_config = config

    def get_session(self, engine_dir):
        serialize_path = engine_dir / 'encoder' / 'rank0.engine'
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self,
                           mel,
                           mel_input_lengths,
                           encoder_downsampling_factor=2):
        if isinstance(mel, list):
            longest_mel = max([f.shape[-1] for f in mel])
            mel = [
                torch.nn.functional.pad(f, (0, longest_mel - f.shape[-1]), mode='constant')
                for f in mel
            ]
            mel = torch.cat(mel, dim=0).type(str_dtype_to_torch("float16")).contiguous()
        bsz, seq_len = mel.shape[0], mel.shape[2]
        position_ids = torch.arange(
            math.ceil(seq_len / encoder_downsampling_factor),
            dtype=torch.int32,
            device=mel.device).expand(bsz, -1).contiguous()
        if self.encoder_config['plugin_config']['remove_input_padding']:
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)
            position_ids = remove_tensor_padding(position_ids,
                                                 mel_input_lengths // encoder_downsampling_factor)
        inputs = OrderedDict()
        inputs['input_features'] = mel
        inputs['input_lengths'] = mel_input_lengths
        inputs['position_ids'] = position_ids

        output_list = [
            TensorInfo('input_features', str_dtype_to_trt(self.dtype),
                       mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       mel_input_lengths.shape),
            TensorInfo('position_ids', str_dtype_to_trt('int32'),
                       inputs['position_ids'].shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        encoder_output = outputs['encoder_output']
        encoder_output_lengths = mel_input_lengths // encoder_downsampling_factor
        return encoder_output, encoder_output_lengths

class EncoderProjector(torch.nn.Module):
    """
    The encoder projector module. It is used to project the encoder outputs to the same dimension as the language model.
    Modified from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/models/projector.py.
    Args:
        encoder_dim (:obj:`int`): The dimension of the encoder outputs.
        llm_dim (:obj:`int`): The dimension of the language model.
        downsample_rate (:obj:`int`, `optional`, defaults to 5): The downsample rate to use.
    """

    def __init__(self, encoder_dim=1280, llm_dim=1536, downsample_rate=8):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * self.downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):

        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class WhisperTRTLLM(nn.Module):

    def __init__(self, engine_dir, llm_dim=1536):
        super().__init__()
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder = WhisperEncoding(engine_dir)
        self.encoder_projector = EncoderProjector(llm_dim=llm_dim)
        self.encoder_projector = self.encoder_projector.half().to("cuda")

    def process_batch(self, mel):
        mel_input_lengths = torch.tensor([f.shape[-1] for f in mel],
                                         dtype=torch.int32,
                                         device='cuda')
        encoder_outputs, encoder_output_lengths = self.encoder.get_audio_features(mel, mel_input_lengths)
        if len(encoder_outputs.shape) == 3:
            speech_features = self.encoder_projector(encoder_outputs)
            speech_features = speech_features.to(torch.float16)
        else:
            assert len(encoder_outputs.shape) == 2
            speech_features = []
            start = 0
            for length in encoder_output_lengths:
                encoder_output = encoder_outputs[start:start + length].unsqueeze(0)
                start += length
                speech_feature = self.encoder_projector(encoder_output).to(torch.float16).squeeze(0)
                speech_features.append(speech_feature)
            assert start == encoder_outputs.shape[0]
        return speech_features

class TritonPythonModel:
    def initialize(self, args):
        device = "cuda"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"
        self.feature_extractor = FeatureExtractor(n_mels=80)
        self.init_model(json.loads(args['model_config'])['parameters'])
        
    def init_model(self, parameters):
        for key,value in parameters.items():
            parameters[key] = value["string_value"]
        engine_dir = parameters["engine_dir"]
        adapter_dir=parameters["adapter_dir"]
        checkpoint = torch.load(
            adapter_dir, map_location="cpu"
        )
        self.llm_dim = checkpoint["encoder_projector.linear1.weight"].shape[0]
        self.model = WhisperTRTLLM(engine_dir, llm_dim=self.llm_dim)
        missing_keys, _ = self.model.load_state_dict(checkpoint, strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        n_mels = int(parameters["n_mels"])
        self.feature_extractor = FeatureExtractor(n_mels=n_mels)

    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses, batch_mel_list = [], []
        for request in requests:
            wav_tensor = pb_utils.get_input_tensor_by_name(request, "WAV")
            wav_len = pb_utils.get_input_tensor_by_name(request, "WAV_LENS").as_numpy().item()
            wav = from_dlpack(wav_tensor.to_dlpack())
            wav = wav[:, :wav_len]
            padding = 3000 if self.llm_dim == 3584 else 0 # WAR: whisper_llm_7b model needs padding
            mel = self.feature_extractor.compute_feature(wav[0].to('cuda'), padding_target_len=padding)
            batch_mel_list.append(mel)

        speech_features_list = self.model.process_batch(batch_mel_list)
        for i in range(len(requests)):
            out_0 = pb_utils.Tensor.from_dlpack("speech_features", to_dlpack(speech_features_list[i].unsqueeze(0)))
            responses.append(pb_utils.InferenceResponse([out_0]))
        return responses