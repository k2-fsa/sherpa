import triton_python_backend_utils as pb_utils
import numpy as np

import json

import torch
import sentencepiece as spm

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

        # Get INPUT configuration

        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out__0")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        self.encoder_dim = encoder_config['dims'][-1]
        
        
        self.init_sentence_piece(self.model_config['parameters'])

    def init_sentence_piece(self, parameters):
        for key,value in parameters.items():
            parameters[key] = value["string_value"]
        self.context_size = int(parameters['context_size'])
        sp = spm.SentencePieceProcessor()
        sp.load(parameters['bpe_model'])
        self.blank_id = sp.piece_to_id("<blk>")
        self.unk_id = sp.piece_to_id("<unk>")
        self.vocab_size = sp.get_piece_size()
        self.sp = sp

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
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out__0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "encoder_out_lens__1")
            batch_encoder_out_list.append(in_0.as_numpy())
            encoder_max_len = max(encoder_max_len, batch_encoder_out_list[-1].shape[1])
            cur_b_lens = in_1.as_numpy()
            batch_encoder_lens_list.append(cur_b_lens)
            cur_batchsize = cur_b_lens.shape[0]
            batchsize_lists.append(cur_batchsize)
            total_seqs += cur_batchsize

      
        encoder_out_array = np.zeros((total_seqs, encoder_max_len, self.encoder_dim),
                                  dtype=self.data_type)
        encoder_out_lens_array = np.zeros(total_seqs, dtype=np.int32)
        st = 0
        for b in batchsize_lists:
            t = batch_encoder_out_list.pop(0)
            encoder_out_array[st:st + b, 0:t.shape[1]] = t
            encoder_out_lens_array[st:st + b] = batch_encoder_lens_list.pop(0)
            st += b

        encoder_out = torch.from_numpy(encoder_out_array)

        encoder_out_lens = torch.from_numpy(encoder_out_lens_array)

        packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
            input=encoder_out,
            lengths=encoder_out_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        pack_batch_size_list = packed_encoder_out.batch_sizes.tolist()
                
        hyps = [[self.blank_id] * self.context_size for _ in range(total_seqs)]
        decoder_input = np.asarray(hyps,dtype=np.int32)
        in_decoder_input_tensor = pb_utils.Tensor("decoder_input__0", decoder_input)

        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['decoder_output__0'],
            inputs=[in_decoder_input_tensor])

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            decoder_out = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'decoder_output__0')
            decoder_out = torch.utils.dlpack.from_dlpack(decoder_out.to_dlpack()).cpu().numpy()
            #decoder_out = decoder_out.as_numpy()

        offset = 0
        
        for batch_size in pack_batch_size_list:
          
            start = offset
            end = offset + batch_size
            current_encoder_out = packed_encoder_out.data[start:end]
            current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1).cpu().numpy()
            # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
 
            offset = end

            decoder_out = decoder_out[:batch_size]
           
            in_joiner_tensor_0 = pb_utils.Tensor("encoder_out__0", current_encoder_out)
            in_joiner_tensor_1 = pb_utils.Tensor("decoder_out__1", np.expand_dims(decoder_out, axis=1))

            inference_request = pb_utils.InferenceRequest(
                model_name='joiner',
                requested_output_names=['logit__0'],
                inputs=[in_joiner_tensor_0, in_joiner_tensor_1])
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(inference_response.error().message())
            else:
                # Extract the output tensors from the inference response.
                logits = pb_utils.get_output_tensor_by_name(inference_response,
                                                                'logit__0')
                logits = torch.utils.dlpack.from_dlpack(logits.to_dlpack()).cpu().numpy()
                
                #logits = logits.as_numpy()
            logits = torch.from_numpy(logits)
            logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
            
            assert logits.ndim == 2, logits.shape
            y = logits.argmax(dim=1).tolist()
            
            emitted = False
            for i, v in enumerate(y):
                if v not in (self.blank_id, self.unk_id):
                    hyps[i].append(v)
                    emitted = True
            if emitted:
                # update decoder output
                decoder_input = [h[-self.context_size:] for h in hyps[:batch_size]]

                decoder_input = np.asarray(decoder_input,dtype=np.int32)

                in_decoder_input_tensor = pb_utils.Tensor("decoder_input__0", decoder_input)

                inference_request = pb_utils.InferenceRequest(
                    model_name='decoder',
                    requested_output_names=['decoder_output__0'],
                    inputs=[in_decoder_input_tensor])

                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(inference_response.error().message())
                else:
                    # Extract the output tensors from the inference response.
                    decoder_out = pb_utils.get_output_tensor_by_name(inference_response,
                                                                    'decoder_output__0')
                    decoder_out = torch.utils.dlpack.from_dlpack(decoder_out.to_dlpack()).cpu().numpy()
                    #decoder_out = decoder_out.as_numpy()

        sorted_ans = [h[self.context_size:] for h in hyps]
        ans = []
        unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
        for i in range(total_seqs):
            ans.append(sorted_ans[unsorted_indices[i]])

        results = []
        for hyp in self.sp.decode(ans):
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