# -*- coding: utf-8 -*-

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack

from .tokenizer import get_tokenizer


def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSCRIPTS")
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        engine_dir = Path(
            self.model_config['parameters']['engine_dir']["string_value"])
        encoder_config = read_config('encoder', engine_dir)
        self.tokenizer = get_tokenizer(
            num_languages=encoder_config['num_languages']
        )
        self.blank = self.tokenizer.encode(
            " ",
            allowed_special=self.tokenizer.special_tokens_set
        )[0]
        self.device = torch.device("cuda")

    def process_batch(self, wav_batch, wav_lens, prompt_id):
        # Convert numpy arrays to torch tensors
        wav_batch = torch.from_numpy(wav_batch).to(self.device)
        wav_tensor = pb_utils.Tensor.from_dlpack(
            "WAV",
            to_dlpack(wav_batch)
        )
        wav_len_tensor = pb_utils.Tensor(
            "WAV_LENS",
            wav_lens.astype(np.int32)
        )

        # Replicate prompt_id for batch size
        batch_size = wav_batch.shape[0]
        prompt_ids = np.tile(prompt_id, (batch_size, 1))
        prompt_ids_tensor = pb_utils.Tensor(
            "DECODER_INPUT_IDS",
            prompt_ids.astype(np.int32)
        )

        infer_request = pb_utils.InferenceRequest(
            model_name="whisper",
            requested_output_names=["OUTPUT_IDS"],
            inputs=[wav_tensor, wav_len_tensor, prompt_ids_tensor]
        )

        inference_response = infer_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())

        output_ids = pb_utils.get_output_tensor_by_name(
            inference_response, "OUTPUT_IDS")
        return output_ids.as_numpy()

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get batch inputs
            text_prefix = pb_utils.get_input_tensor_by_name(
                request, "TEXT_PREFIX").as_numpy()
            wav_batch = pb_utils.get_input_tensor_by_name(
                request, "WAV").as_numpy()
            wav_lens = pb_utils.get_input_tensor_by_name(
                request, "WAV_LENS").as_numpy()

            # Use the same text_prefix for all items in the request
            prefix = text_prefix[0][0].decode('utf-8')
            if prefix == "":
                prefix = (
                    "<|startoftranscript|><|ko|><|transcribe|><|notimestamps|>"
                )
            prompt_id = self.tokenizer.encode(
                prefix,
                allowed_special=self.tokenizer.special_tokens_set
            )

            # Process the entire batch
            output_ids = self.process_batch(wav_batch, wav_lens, prompt_id)

            # Decode outputs for each item in batch
            transcripts = []

            # Handle case where output_ids is 3-dimensional
            # ([batch_size, beam_size, seq_len])
            if len(output_ids.shape) == 3:
                output_ids = output_ids[:, 0, :]  # Remove beam_size dimension

            for output_id in output_ids:
                token_list = output_id.tolist()
                s = self.tokenizer.decode(token_list)
                s = re.sub(r'<\|.*?\|>', '', s)
                transcripts.append(s)

            # Create response tensor
            out0 = pb_utils.Tensor(
                "TRANSCRIPTS",
                np.array(transcripts).astype(self.out0_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')
