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
from torch.utils.dlpack import to_dlpack, from_dlpack
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kaldifeat
import _kaldifeat
from typing import List
import json
import math
from typing import Optional, Tuple

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.
    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


class FrameReducer(torch.nn.Module):
    """The encoder output is first used to calculate
    the CTC posterior probability; then for each output frame,
    if its blank posterior is bigger than some thresholds,
    it will be simply discarded from the encoder output.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        ctc_output: torch.Tensor,
        y_lens: Optional[torch.Tensor] = None,
        blank_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
              The shared encoder output with shape [N, T, C].
            x_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `x` before padding.
            ctc_output:
              The CTC output with shape [N, T, vocab_size].
            y_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `y` before padding.
            blank_id:
              The blank id of ctc_output.
        Returns:
            out:
              The frame reduced encoder output with shape [N, T', C].
            out_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `out` before padding.
        """
        N, T, C = x.size()

        padding_mask = make_pad_mask(x_lens)
        left = ctc_output[:, :, blank_id] < math.log(0.9)
        non_blank_mask = torch.logical_and(left.to(x.device), (~padding_mask))
        #non_blank_mask = left * (~padding_mask)

        out_lens = non_blank_mask.sum(dim=1).to(x.device)
        max_len = out_lens.max()

        pad_lens_list = (
            torch.full_like(
                out_lens,
                max_len.item(),
                device=x.device,
            )
            - out_lens
        )
        max_pad_len = pad_lens_list.max()

        out = F.pad(x, (0, 0, 0, max_pad_len))

        valid_pad_mask = ~make_pad_mask(pad_lens_list)
        total_valid_mask = torch.concat([non_blank_mask, valid_pad_mask], dim=1)

        out = out[total_valid_mask].reshape(N, -1, C)

        return out, out_lens


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

        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Get INPUT configuration
        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "x")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])
        if self.data_type == np.float32:
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.float16

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "out")
        # Convert Triton types to numpy types
        output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        if output0_dtype == np.float32:
            self.output0_dtype = torch.float32
        else:
            self.output0_dtype = torch.float16

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "out_lens")
        # Convert Triton types to numpy types
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])

        params = self.model_config['parameters']
        self.frame_reducer= FrameReducer()


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
        batch_encoder_out_list, batch_encoder_lens_list, batch_ctc_out_list = [], [], []
        batchsize_lists = []
        total_seqs = 0
        encoder_max_len = 0
        batch_masks = []

        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "x")
            in_1 = pb_utils.get_input_tensor_by_name(request, "x_lens")
            
            batch_encoder_out_list.append(from_dlpack(in_0.to_dlpack()))
            encoder_max_len = max(encoder_max_len, batch_encoder_out_list[-1].shape[1])

            cur_b_lens = from_dlpack(in_1.to_dlpack())

            batch_encoder_lens_list.append(cur_b_lens)
            cur_batchsize = cur_b_lens.shape[0]
            batchsize_lists.append(cur_batchsize)
            total_seqs += cur_batchsize

        encoder_out = torch.zeros((total_seqs, encoder_max_len, 384),
                                  dtype=self.torch_dtype, device=self.device)
        encoder_out_lens = torch.zeros(total_seqs, dtype=torch.int64, device=self.device)
        st = 0


        for b in batchsize_lists:
            t = batch_encoder_out_list.pop(0)
            encoder_out[st:st + b, 0:t.shape[1]] = t
            encoder_out_lens[st:st + b] = batch_encoder_lens_list.pop(0)

            st += b

        in_tensor_0 = pb_utils.Tensor.from_dlpack("encoder_out", to_dlpack(encoder_out))

        inference_request = pb_utils.InferenceRequest(
            model_name='ctc_model',
            requested_output_names=['ctc_output'],
            inputs=[in_tensor_0])

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            ctc_out = pb_utils.get_output_tensor_by_name(inference_response, 'ctc_output')
            ctc_out = from_dlpack(ctc_out.to_dlpack())

        in_tensor_0 = pb_utils.Tensor.from_dlpack("lconv_input", to_dlpack(encoder_out))
        in_tensor_1 = pb_utils.Tensor.from_dlpack("lconv_input_lens", to_dlpack(encoder_out_lens.unsqueeze(-1)))

        input_tensors = [in_tensor_0, in_tensor_1]
        inference_request = pb_utils.InferenceRequest(
            model_name='lconv',
            requested_output_names=['lconv_out'],
            inputs=input_tensors)

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            lconv_out = pb_utils.get_output_tensor_by_name(inference_response, 'lconv_out')
            lconv_out = from_dlpack(lconv_out.to_dlpack())

        out, out_lens = self.frame_reducer(encoder_out, encoder_out_lens, ctc_out)

        st = 0
        responses = []
        for b in batchsize_lists:
            speech = out[st:st+b]
            speech_lengths = out_lens[st:st+b]
            out0 = pb_utils.Tensor.from_dlpack("out", to_dlpack(speech))
            out1 = pb_utils.Tensor.from_dlpack("out_lens",
                                               to_dlpack(speech_lengths))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0, out1])
            responses.append(inference_response)
        return responses
