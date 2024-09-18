
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

def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=0):
    if input_tensor.dim() == 2:
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] != pad_value
        ), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    elif input_tensor.dim() == 3:
        # Audio tensor case: batch, seq_len, feature_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"
        batch_size, seq_len, feature_len = input_tensor.shape

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(batch_size):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length, :])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)

    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

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
                           mel):
        mel_input_lengths = torch.tensor(
            [mel.shape[2] for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device)
        if self.encoder_config['plugin_config']['remove_input_padding']:
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)

        inputs = OrderedDict()
        inputs['input_features'] = mel
        inputs['input_lengths'] = mel_input_lengths

        output_list = [
            TensorInfo('input_features', str_dtype_to_trt(self.dtype),
                       mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       mel_input_lengths.shape)
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
        encoder_output_lengths = mel_input_lengths // 2

        return encoder_output

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

    def __init__(self, engine_dir):
        super().__init__()
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder = WhisperEncoding(engine_dir)
        self.encoder_projector = EncoderProjector()
        self.encoder_projector = self.encoder_projector.half().to("cuda")

    def process_batch(self, mel):
        encoder_outputs = self.encoder.get_audio_features(mel)
        speech_features = self.encoder_projector(encoder_outputs)
        speech_features = speech_features.to(torch.float16)
        return speech_features

class TritonPythonModel:
    def initialize(self, args):
        device = "cuda"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"

        self.init_model(json.loads(args['model_config'])['parameters'])
        
    def init_model(self, parameters):
        for key,value in parameters.items():
            parameters[key] = value["string_value"]
        engine_dir = parameters["engine_dir"]
        adapter_dir=parameters["adapter_dir"]
        checkpoint = torch.load(
            adapter_dir, map_location="cpu"
        )
        self.model = WhisperTRTLLM(engine_dir)
        missing_keys, _ = self.model.load_state_dict(checkpoint, strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"

    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses, batch_mel_list = [], []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "mel")
            # assert not in_0.is_cpu()
            batch_mel_list.append(from_dlpack(in_0.to_dlpack()))
        # concatenate all mel tensors in the batch
        batch_mel = torch.cat(batch_mel_list, dim=0).to(self.device)
        speech_features = self.model.process_batch(batch_mel)
        for i in range(len(requests)):
            out_0 = pb_utils.Tensor.from_dlpack("speech_features", to_dlpack(speech_features[i].unsqueeze(0)))
            responses.append(pb_utils.InferenceResponse([out_0]))
        return responses