# -*- coding: utf-8 -*- 
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import torch
from torch.utils.dlpack import to_dlpack
import re
from .tokenizer import get_tokenizer
from collections import OrderedDict
from pathlib import Path

def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSCRIPTS")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        encoder_config = read_config('encoder', Path(self.model_config['parameters']['engine_dir']["string_value"]))
        self.tokenizer = get_tokenizer(num_languages=encoder_config['num_languages'])
        self.blank = self.tokenizer.encode(" ", allowed_special=self.tokenizer.special_tokens_set)[0]
        self.device = torch.device("cuda")

    def process_batch(self, wav, wav_len, prompt_id):
        wav = torch.from_numpy(wav[0]).to(self.device)
        wav_tensor = pb_utils.Tensor.from_dlpack("WAV", to_dlpack(wav.unsqueeze(0)))
        wav_len_tensor = pb_utils.Tensor("WAV_LENS", np.array([[wav_len]], np.int32))
        prompt_id = torch.tensor(prompt_id).unsqueeze(0)

        prompt_id = pb_utils.Tensor("DECODER_INPUT_IDS", prompt_id.numpy().astype(np.int32))
        infer_request = pb_utils.InferenceRequest(
            model_name="whisper",
            requested_output_names=["OUTPUT_IDS"],
            inputs=[wav_tensor, wav_len_tensor, prompt_id]
        )
        inference_response = infer_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            output_ids = pb_utils.get_output_tensor_by_name(inference_response, "OUTPUT_IDS")
            return output_ids.as_numpy()
        
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        responses = []
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "TEXT_PREFIX")
            prompt_ids = in_0.as_numpy().tolist()
            prompt_ids = prompt_ids[0][0].decode('utf-8')
            if prompt_ids == "":
                prompt_ids = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
            prompt_id = self.tokenizer.encode(prompt_ids, allowed_special=self.tokenizer.special_tokens_set)

            wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
            assert wav.shape[0] == 1, "Only support batch size 1 for now"
            wav_len = pb_utils.get_input_tensor_by_name(request, "WAV_LENS").as_numpy()
            wav_len = wav_len.item()

            output_ids = self.process_batch(wav, wav_len, prompt_id)
            s = self.tokenizer.decode(output_ids)
            s = re.sub(r'<\|.*?\|>', '', s)
            sentence = np.array([s])
            out0 = pb_utils.Tensor("TRANSCRIPTS", sentence.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
