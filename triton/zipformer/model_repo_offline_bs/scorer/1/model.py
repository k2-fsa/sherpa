# -*- coding: utf-8 -*- 
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import triton_python_backend_utils as pb_utils
import numpy as np

import json

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import sentencepiece as spm
from icefall.lexicon import Lexicon

from search import greedy_search

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
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        model_instance_kind = args['model_instance_kind']
        model_instance_device_id = args['model_instance_device_id']
        if model_instance_kind == 'GPU':
            self.device = f'cuda:{model_instance_device_id}'
        else:
            self.device= 'cpu'

        # Get INPUT configuration
        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])
        if self.data_type == np.float32:
            self.torch_dtype = torch.float32
        else:
            assert self.data_type == np.float16
            self.torch_dtype = torch.float16

        self.encoder_dim = encoder_config['dims'][-1]
        
        
        self.init_parameters(self.model_config['parameters'])

    def init_parameters(self, parameters):
        for key,value in parameters.items():
            parameters[key] = value["string_value"]
        self.context_size = int(parameters['context_size'])
        self.decoding_method = parameters['decoding_method']
        if 'bpe' in parameters['tokenizer_file']:
            sp = spm.SentencePieceProcessor()
            sp.load(parameters['tokenizer_file'])
            self.blank_id = sp.piece_to_id("<blk>")
            self.unk_id = sp.piece_to_id("<unk>")
            self.vocab_size = sp.get_piece_size()
            self.tokenizer = sp
        else:
            assert 'char' in parameters['tokenizer_file']
            lexicon = Lexicon(parameters['tokenizer_file'])
            self.unk_id = lexicon.token_table["<unk>"]
            self.blank_id = lexicon.token_table["<blk>"]
            self.vocab_size = max(lexicon.tokens) + 1
            self.tokenizer = lexicon

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

        batch_encoder_out_list, batch_encoder_lens_list = [], []    
        batchsize_lists = []
        total_seqs = 0
        encoder_max_len = 0

        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            in_1 = pb_utils.get_input_tensor_by_name(request, "encoder_out_lens")
            assert not in_0.is_cpu()
            batch_encoder_out_list.append(from_dlpack(in_0.to_dlpack()))
            encoder_max_len = max(encoder_max_len, batch_encoder_out_list[-1].shape[1])
            cur_b_lens = from_dlpack(in_1.to_dlpack())
            batch_encoder_lens_list.append(cur_b_lens)
            cur_batchsize = cur_b_lens.shape[0]
            batchsize_lists.append(cur_batchsize)
            total_seqs += cur_batchsize

        encoder_out = torch.zeros((total_seqs, encoder_max_len, self.encoder_dim),
                                  dtype=self.torch_dtype, device=self.device)
        encoder_out_lens = torch.zeros(total_seqs, dtype=torch.int64)
        st = 0
    
        for b in batchsize_lists:
            t = batch_encoder_out_list.pop(0)
            encoder_out[st:st + b, 0:t.shape[1]] = t
            encoder_out_lens[st:st + b] = batch_encoder_lens_list.pop(0)
            st += b

        if self.decoding_method == 'greedy_search':
            ans = greedy_search(encoder_out, encoder_out_lens, self.context_size, self.unk_id, self.blank_id)
        else:
            raise NotImplementedError
        results = []
        if hasattr(self.tokenizer, 'token_table'):
            for i in range(len(ans)):
                results.append([self.tokenizer.token_table[idx] for idx in ans[i]])
        else:
            for hyp in self.tokenizer.decode(ans):
                results.append(hyp.split())
        st = 0
        responses = []
        for b in batchsize_lists:
            sents = np.array(results[st:st + b])
            out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)
            st += b
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
