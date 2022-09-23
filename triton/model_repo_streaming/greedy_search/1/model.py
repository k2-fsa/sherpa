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
            model_config, "encoder_out")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        self.encoder_dim = encoder_config['dims'][-1]
        
        
        self.init_sentence_piece(self.model_config['parameters'])

        # use to record every sequence state
        self.seq_states = {}
        print("Finish Init")

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


    def forward_joiner(self, cur_encoder_out, decoder_out):
        in_joiner_tensor_0 = pb_utils.Tensor("encoder_out", cur_encoder_out.cpu().numpy())
        in_joiner_tensor_1 = pb_utils.Tensor("decoder_out", decoder_out.cpu().numpy())

        inference_request = pb_utils.InferenceRequest(
            model_name='joiner',
            requested_output_names=['logit'],
            inputs=[in_joiner_tensor_0, in_joiner_tensor_1])
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


    def forward_decoder(self,hyps):
        decoder_input = [h[-self.context_size:] for h in hyps]

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
            decoder_out = torch.utils.dlpack.from_dlpack(decoder_out.to_dlpack()).cpu()
            assert len(decoder_out.shape)==3, decoder_out.shape
            decoder_out = decoder_out.squeeze(1)
            return decoder_out
            

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
        batch_idx = 0
        encoder_max_len = 0


        batch_idx2_corrid = {}
    
        hyps_list = []
        decoder_out_list = []
        end_idx = set()

        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            in_1 = pb_utils.get_input_tensor_by_name(request, "encoder_out_lens")

            # TODO: directly use torch tensor from_dlpack(in_0.to_dlpack())
            batch_encoder_out_list.append(in_0.as_numpy())

            assert batch_encoder_out_list[-1].shape[0] == 1            
            encoder_max_len = max(encoder_max_len, batch_encoder_out_list[-1].shape[1])

            cur_b_lens = in_1.as_numpy()
            batch_encoder_lens_list.append(cur_b_lens)

            assert encoder_max_len == cur_b_lens[0]

            # For streaming ASR, assert each request sent from client has batch size 1.
            

            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]

            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]

            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]

            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]

            if start and ready:
                # intialize states
                init_hyp = [self.blank_id] * self.context_size
                init_decoder_out = self.forward_decoder([init_hyp])   
                self.seq_states[corrid] = [init_hyp, init_decoder_out]

            if end and ready:
                end_idx.add(batch_idx)
    
            if ready:
                hyp, decoder_out = self.seq_states[corrid]
                batch_idx2_corrid[batch_idx] = corrid
                hyps_list.append(hyp)
                decoder_out_list.append(decoder_out)

            batch_idx += 1
                      
        encoder_out_array = np.zeros((batch_idx, encoder_max_len, self.encoder_dim),
                                  dtype=self.data_type)
        encoder_out_lens_array = np.zeros(batch_idx, dtype=np.int32)


        for i, t in enumerate(batch_encoder_out_list):
            
            encoder_out_array[i, 0:t.shape[1]] = t
            encoder_out_lens_array[i] = batch_encoder_lens_list[i]
    
        encoder_out = torch.from_numpy(encoder_out_array)
        encoder_out_lens = torch.from_numpy(encoder_out_lens_array)

        decoder_out = torch.cat(decoder_out_list, dim=0)

    

        assert encoder_out.shape[0] == decoder_out.shape[0]

        

        for t in range(encoder_out.shape[1]):
            cur_encoder_out = encoder_out[:,t]
            logits = self.forward_joiner(cur_encoder_out, decoder_out)

            assert logits.ndim == 2, logits.shape
            y = logits.argmax(dim=1).tolist()
            
            emitted = False
            for i, v in enumerate(y):
                if v not in (self.blank_id, self.unk_id):
                    hyps_list[i].append(v)
                    emitted = True
            if emitted:
                decoder_out = self.forward_decoder(hyps_list)

        responses = []
        for i in range(len(hyps_list)):
            hyp = hyps_list[i][self.context_size:]
            sent = self.sp.decode(hyp).split()
            sent = np.array(sent)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", sent.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
    
            corr = batch_idx2_corrid[i]
            if i in end_idx:
                del self.seq_states[corr]
            else:
                self.seq_states[corr] = [hyps_list[i], decoder_out[i].unsqueeze(0)]
                    
        assert len(requests) == len(responses)
        return responses
    
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')