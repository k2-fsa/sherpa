# -*- coding: utf-8 -*- 
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from .tokenizer import get_tokenizer

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
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.tokenizer = get_tokenizer()
        self.sot = self.tokenizer.encode("<|startoftranscript|>", allowed_special=self.tokenizer.special_tokens_set)[0]
        self.eot = self.tokenizer.encode("<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set)[0]
        self.translate = self.tokenizer.encode("<|translate|>", allowed_special=self.tokenizer.special_tokens_set)[0]
        self.no_timestamps = self.tokenizer.encode("<|notimestamps|>", allowed_special=self.tokenizer.special_tokens_set)[0]
        self.no_speech = self.tokenizer.encode("<|nospeech|>", allowed_special=self.tokenizer.special_tokens_set)[0]
        self.blank = self.tokenizer.encode(" ", allowed_special=self.tokenizer.special_tokens_set)[0]

        self.init_parameters(self.model_config['parameters'])

    def init_parameters(self, parameters):
        for key,value in parameters.items():
            parameters[key] = value["string_value"]
        self.n_text_layer = int(parameters["n_text_layer"])
        self.n_text_ctx = int(parameters["n_text_ctx"])
        self.n_text_state = int(parameters["n_text_state"])

    def forward_decoder(self, tokens, n_layer_self_k_cache, n_layer_self_v_cache, n_layer_cross_k, n_layer_cross_v, offset):

        in_tokens = pb_utils.Tensor.from_dlpack("tokens", to_dlpack(tokens))
        in_n_layer_self_k_cache = pb_utils.Tensor.from_dlpack("in_n_layer_self_k_cache", to_dlpack(n_layer_self_k_cache))
        in_n_layer_self_v_cache = pb_utils.Tensor.from_dlpack("in_n_layer_self_v_cache", to_dlpack(n_layer_self_v_cache))
        in_n_layer_cross_k = pb_utils.Tensor.from_dlpack("n_layer_cross_k", to_dlpack(n_layer_cross_k))
        in_n_layer_cross_v = pb_utils.Tensor.from_dlpack("n_layer_cross_v", to_dlpack(n_layer_cross_v))
        in_offset = pb_utils.Tensor.from_dlpack("offset", to_dlpack(offset))
    
        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['logits', 'out_n_layer_self_k_cache', 'out_n_layer_self_v_cache'],
            inputs=[in_tokens, in_n_layer_self_k_cache, in_n_layer_self_v_cache, in_n_layer_cross_k, in_n_layer_cross_v, in_offset])
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            logits = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'logits')
            logits = torch.utils.dlpack.from_dlpack(logits.to_dlpack())
            next_n_layer_self_k_cache = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'out_n_layer_self_k_cache')
            next_n_layer_self_k_cache = torch.utils.dlpack.from_dlpack(next_n_layer_self_k_cache.to_dlpack())
            next_n_layer_self_v_cache = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'out_n_layer_self_v_cache')
            next_n_layer_self_v_cache = torch.utils.dlpack.from_dlpack(next_n_layer_self_v_cache.to_dlpack())

            return logits, next_n_layer_self_k_cache, next_n_layer_self_v_cache

    def get_init_self_attention_cache(self, batch_size):
        n_layer_self_k_cache = torch.zeros(
            batch_size,
            self.n_text_layer,
            self.n_text_ctx,
            self.n_text_state,
        )
        n_layer_self_v_cache = torch.zeros(
            batch_size,
            self.n_text_layer,
            self.n_text_ctx,
            self.n_text_state,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache        
    
    def suppress_tokens(self, logits, is_initial: bool) -> None:
        # suppress blank
        if is_initial:
            logits[:,self.eot] = float("-inf")
            logits[:,self.blank] = float("-inf")

        # suppress <|notimestamps|>
        logits[:,self.no_timestamps] = float("-inf")

        logits[:,self.sot] = float("-inf")
        logits[:,self.no_speech] = float("-inf")

        # logits is changed in-place
        logits[:,self.translate] = float("-inf")

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
        n_layer_cross_k_list, n_layer_cross_v_list, text_prefix_list = [], [], []
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "n_layer_cross_k")
            in_1 = pb_utils.get_input_tensor_by_name(request, "n_layer_cross_v")
            in_2 = pb_utils.get_input_tensor_by_name(request, "text_prefix")
            assert not in_0.is_cpu()
            assert not in_1.is_cpu()
            n_layer_cross_k_list.append(from_dlpack(in_0.to_dlpack()))
            n_layer_cross_v_list.append(from_dlpack(in_1.to_dlpack()))
            text_prefix_list.append(in_2.as_numpy().tolist())
        # concat tensors in batch dimension
        n_layer_cross_k = torch.cat(n_layer_cross_k_list, dim=0)
        n_layer_cross_v = torch.cat(n_layer_cross_v_list, dim=0)

        prompt_ids = []
        for text_prefix in text_prefix_list:
            text_prefix = text_prefix[0][0].decode('utf-8')
            if text_prefix == "":
                text_prefix = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
            prompt_id = self.tokenizer.encode(text_prefix, allowed_special=self.tokenizer.special_tokens_set)
            # convert prompt_id to tensor, tensor shape is [Seq]
            prompt_id = torch.tensor(prompt_id)
            prompt_ids.append(prompt_id)
        # convert prompt_ids to tensor, tensor shape is [Batch, Seq], left padding with self.blank
        tokens = torch.nn.utils.rnn.pad_sequence(prompt_ids, batch_first=True, padding_value=self.blank)
        tokens = tokens.to(n_layer_cross_k.device)

        n_layer_self_k_cache, n_layer_self_v_cache = self.get_init_self_attention_cache(n_layer_cross_k.shape[0])
        n_layer_self_k_cache = n_layer_self_k_cache.to(n_layer_cross_k.device)
        n_layer_self_v_cache = n_layer_self_v_cache.to(n_layer_cross_k.device)

        # make offset tensor with shape [Batch, 1]
        offset = torch.tensor([[0]] * n_layer_cross_k.shape[0], dtype=torch.long, device=n_layer_cross_k.device)

        logits, n_layer_self_k_cache, n_layer_self_v_cache = self.forward_decoder(
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        )

        logits = logits[:, -1, :]
        self.suppress_tokens(logits, is_initial=True)
        # update offset for next inference
        offset += tokens.shape[1]
        max_token_ids = torch.argmax(logits, dim=-1)

        results = [[] for _ in range(n_layer_cross_k.shape[0])]
        ongoing_index = [i for i in range(n_layer_cross_k.shape[0])]

        for i in range(self.n_text_ctx):
            assert len(ongoing_index) == len(max_token_ids)
            tokens = max_token_ids.clone()
            selected_relative_index = []
            for j, max_token_id in enumerate(max_token_ids):
                if max_token_id != self.eot:
                    selected_relative_index.append(j)
                    results[ongoing_index[j]].append(max_token_id.item())
            if len(selected_relative_index) == 0:
                break
            tmp_index = [ongoing_index[i] for i in range(len(ongoing_index)) if i in selected_relative_index]
            ongoing_index = tmp_index
            offset = offset[selected_relative_index]
            n_layer_self_k_cache = n_layer_self_k_cache[selected_relative_index]
            n_layer_self_v_cache = n_layer_self_v_cache[selected_relative_index]
            n_layer_cross_k = n_layer_cross_k[selected_relative_index]
            n_layer_cross_v = n_layer_cross_v[selected_relative_index]
            tokens = tokens[selected_relative_index]
            # tokens shape is [New_batch, 1], device is same as logits
            assert tokens.shape[0] == len(ongoing_index)
            tokens = tokens.unsqueeze(1)

            logits, n_layer_self_k_cache, n_layer_self_v_cache = self.forward_decoder(
                tokens=tokens,
                n_layer_self_k_cache=n_layer_self_k_cache,
                n_layer_self_v_cache=n_layer_self_v_cache,
                n_layer_cross_k=n_layer_cross_k,
                n_layer_cross_v=n_layer_cross_v,
                offset=offset,
            )
            logits = logits[:, -1]
            self.suppress_tokens(logits, is_initial=False)
            max_token_ids = logits.argmax(dim=-1)
            offset += 1

        responses = []
        for result in results:
            s = self.tokenizer.decode(result)
            sentence = np.array([s])
            out0 = pb_utils.Tensor("OUTPUT0", sentence.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
