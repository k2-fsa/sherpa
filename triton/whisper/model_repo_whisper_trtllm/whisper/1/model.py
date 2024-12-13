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
from pathlib import Path

import torch
import triton_python_backend_utils as pb_utils
from tensorrt_llm.bindings import GptJsonConfig
from tensorrt_llm.runtime import ModelRunnerCpp
from torch.utils.dlpack import from_dlpack

from .fbank import FeatureExtractor


class TritonPythonModel:
    def initialize(self, args):
        parameters = json.loads(args['model_config'])['parameters']
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        engine_dir = parameters["engine_dir"]
        config_path = Path(engine_dir) / 'decoder' / 'config.json'
        json_config = GptJsonConfig.parse_file(config_path)
        assert json_config.model_config.supports_inflight_batching
        runner_kwargs = dict(
            engine_dir=engine_dir,
            is_enc_dec=True,
            max_batch_size=64,
            max_input_len=3000,
            max_output_len=96,
            max_beam_width=1,
            debug_mode=False,
            kv_cache_free_gpu_memory_fraction=0.5,
        )
        self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.feature_extractor = FeatureExtractor(
            n_mels=int(parameters["n_mels"])
        )
        self.zero_pad = parameters["zero_pad"] == "true"
        self.eot_id = 50257

    def execute(self, requests):
        responses = []

        for request in requests:
            wav_tensor = pb_utils.get_input_tensor_by_name(request, "WAV")
            wav_lens = pb_utils.get_input_tensor_by_name(
                request, "WAV_LENS").as_numpy()
            prompt_ids = pb_utils.get_input_tensor_by_name(
                request, "DECODER_INPUT_IDS").as_numpy()

            # Move WAV data to GPU
            wav = from_dlpack(wav_tensor.to_dlpack())
            batch_size = wav.shape[0]

            padding = 0 if self.zero_pad else 3000
            batch_mel_list = []

            # Batch processing for each sample in the batch
            for i in range(batch_size):
                wav_i = wav[i:i+1, :int(wav_lens[i].item())]
                mel = self.feature_extractor.compute_feature(
                    wav_i[0].to('cuda'),
                    padding_target_len=padding
                ).transpose(1, 2)
                batch_mel_list.append(mel.squeeze(0))

            # Move prompt IDs to GPU
            decoder_input_ids = torch.tensor(
                prompt_ids, dtype=torch.int32, device='cuda')

            # Calculate mel lengths
            mel_input_lengths = torch.tensor(
                [mel.shape[0] for mel in batch_mel_list],
                dtype=torch.int32,
                device='cuda'
            )

            # Run batch inference
            outputs = self.model_runner_cpp.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=batch_mel_list,
                encoder_output_lengths=mel_input_lengths // 2,
                max_new_tokens=96,
                end_id=self.eot_id,
                pad_id=self.eot_id,
                num_beams=1,
                output_sequence_lengths=True,
                return_dict=True
            )
            torch.cuda.synchronize()

            # Process outputs
            output_ids = outputs['output_ids'].cpu().numpy()

            # Create response for the request
            response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("OUTPUT_IDS", output_ids)
            ])
            responses.append(response)

        return responses
