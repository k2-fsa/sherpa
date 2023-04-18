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

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

def forward_joiner(cur_encoder_out, decoder_out):
    in_joiner_tensor_0 = pb_utils.Tensor.from_dlpack("encoder_out", to_dlpack(cur_encoder_out))
    in_joiner_tensor_1 = pb_utils.Tensor.from_dlpack("decoder_out", to_dlpack(decoder_out.squeeze(1)))

    inference_request = pb_utils.InferenceRequest(
        model_name='joiner_encoder_proj',
        requested_output_names=['projected_encoder_out'],
        inputs=[in_joiner_tensor_0])
    inference_response = inference_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())
    else:
        # Extract the output tensors from the inference response.
        proj_encoder_out = pb_utils.get_output_tensor_by_name(inference_response,
                                                        'projected_encoder_out')
        proj_encoder_out = from_dlpack(proj_encoder_out.to_dlpack())

    inference_request = pb_utils.InferenceRequest(
        model_name='joiner_decoder_proj',
        requested_output_names=['projected_decoder_out'],
        inputs=[in_joiner_tensor_1])
    inference_response = inference_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())
    else:
        # Extract the output tensors from the inference response.
        proj_decoder_out = pb_utils.get_output_tensor_by_name(inference_response,
                                                        'projected_decoder_out')
        proj_decoder_out = from_dlpack(proj_decoder_out.to_dlpack())


    proj_encoder = pb_utils.Tensor.from_dlpack("encoder_out", to_dlpack(proj_encoder_out))
    proj_decoder = pb_utils.Tensor.from_dlpack("decoder_out", to_dlpack(proj_decoder_out))

    inference_request = pb_utils.InferenceRequest(
        model_name='joiner',
        requested_output_names=['logit'],
        inputs=[proj_encoder, proj_decoder])
    inference_response = inference_request.exec()

    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())
    else:
        # Extract the output tensors from the inference response.
        logits = pb_utils.get_output_tensor_by_name(inference_response,
                                                        'logit')
        logits = torch.utils.dlpack.from_dlpack(logits.to_dlpack()).cpu()
        assert len(logits.shape) == 2, logits.shape
        return logits

def forward_decoder(hyps, context_size):
    decoder_input = [h[-context_size:] for h in hyps]

    decoder_input = np.asarray(decoder_input,dtype=np.int64)

    in_decoder_input_tensor = pb_utils.Tensor("y", decoder_input)

    inference_request = pb_utils.InferenceRequest(
        model_name='decoder',
        requested_output_names=['decoder_out'],
        inputs=[in_decoder_input_tensor])

    inference_response = inference_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())
    else:
        # Extract the output tensors from the inference response.
        decoder_out = pb_utils.get_output_tensor_by_name(inference_response,
                                                        'decoder_out')
        decoder_out = from_dlpack(decoder_out.to_dlpack())
        return decoder_out


def greedy_search(encoder_out, encoder_out_lens, context_size, unk_id, blank_id):
    
    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False
    )

    pack_batch_size_list = packed_encoder_out.batch_sizes.tolist()
            
    hyps = [[blank_id] * context_size for _ in range(encoder_out.shape[0])]
    decoder_out = forward_decoder(hyps, context_size)

    offset = 0
    for batch_size in pack_batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = packed_encoder_out.data[start:end]

        offset = end
    
        decoder_out = decoder_out[:batch_size]

        logits = forward_joiner(current_encoder_out, decoder_out)

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                emitted = True
        if emitted:
            decoder_out = forward_decoder(hyps[:batch_size], context_size)


    sorted_ans = [h[context_size:] for h in hyps]

    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(encoder_out.shape[0]):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans
