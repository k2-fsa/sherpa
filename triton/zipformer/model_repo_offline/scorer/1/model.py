# -*- coding: utf-8 -*- 
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import sentencepiece as spm
from icefall.lexicon import Lexicon
from typing import List, Union
import k2

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
        if self.decoding_method == 'fast_beam_search':
            # parameters for fast beam search
            self.beam = int(self.model_config['parameters']['beam'])
            self.max_contexts = int(self.model_config['parameters']['max_contexts'])
            self.max_states = int(self.model_config['parameters']['max_states'])
            self.temperature = float(self.model_config['parameters']['temperature'])
            # Support fast beam search one best currently
            self.decoding_graph = k2.trivial_graph(
                    self.vocab_size - 1, device=self.device
                )

    def forward_joiner(self, cur_encoder_out, decoder_out):
        in_joiner_tensor_0 = pb_utils.Tensor.from_dlpack("encoder_out", to_dlpack(cur_encoder_out))
        in_joiner_tensor_1 = pb_utils.Tensor.from_dlpack("decoder_out", to_dlpack(decoder_out.squeeze(1)))
    
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
    
    def forward_decoder(self, hyps):
        decoder_input = np.asarray(hyps,dtype=np.int64)
    
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
    
    
    def greedy_search(self, encoder_out, encoder_out_lens):
        
        packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
            input=encoder_out,
            lengths=encoder_out_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
    
        pack_batch_size_list = packed_encoder_out.batch_sizes.tolist()
                
        hyps = [[self.blank_id] * self.context_size for _ in range(encoder_out.shape[0])]
        contexts = [h[-self.context_size:] for h in hyps]
        decoder_out = self.forward_decoder(contexts)
    
        offset = 0
        for batch_size in pack_batch_size_list:
            start = offset
            end = offset + batch_size
            current_encoder_out = packed_encoder_out.data[start:end]

            offset = end
        
            decoder_out = decoder_out[:batch_size]
    
            logits = self.forward_joiner(current_encoder_out, decoder_out)
    
            assert logits.ndim == 2, logits.shape
            y = logits.argmax(dim=1).tolist()
            
            emitted = False
            for i, v in enumerate(y):
                if v not in (self.blank_id, self.unk_id):
                    hyps[i].append(v)
                    emitted = True
            if emitted:
                hyps = hyps[:batch_size]
                contexts = [h[-self.context_size:] for h in hyps]
                decoder_out = self.forward_decoder(contexts)
    
    
        sorted_ans = [h[self.context_size:] for h in hyps]
    
        ans = []
        unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
        for i in range(encoder_out.shape[0]):
            ans.append(sorted_ans[unsorted_indices[i]])
    
        return ans
    
    # From k2 utils.py
    def get_texts(self, 
        best_paths: k2.Fsa, return_ragged: bool = False
    ) -> Union[List[List[int]], k2.RaggedTensor]:
        """Extract the texts (as word IDs) from the best-path FSAs.
        Args:
          best_paths:
            A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
            containing multiple FSAs, which is expected to be the result
            of k2.shortest_path (otherwise the returned values won't
            be meaningful).
          return_ragged:
            True to return a ragged tensor with two axes [utt][word_id].
            False to return a list-of-list word IDs.
        Returns:
          Returns a list of lists of int, containing the label sequences we
          decoded.
        """
        if isinstance(best_paths.aux_labels, k2.RaggedTensor):
            # remove 0's and -1's.
            aux_labels = best_paths.aux_labels.remove_values_leq(0)
            # TODO: change arcs.shape() to arcs.shape
            aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)
    
            # remove the states and arcs axes.
            aux_shape = aux_shape.remove_axis(1)
            aux_shape = aux_shape.remove_axis(1)
            aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
        else:
            # remove axis corresponding to states.
            aux_shape = best_paths.arcs.shape().remove_axis(1)
            aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
            # remove 0's and -1's.
            aux_labels = aux_labels.remove_values_leq(0)
    
        assert aux_labels.num_axes == 2
        if return_ragged:
            return aux_labels
        else:
            return aux_labels.tolist()
    
    def fast_beam_search(self, encoder_out, encoder_out_lens):
        B, T, C = encoder_out.shape
      
        config = k2.RnntDecodingConfig(
            vocab_size=self.vocab_size,
            decoder_history_len=self.context_size,
            beam=self.beam,
            max_contexts=self.max_contexts,
            max_states=self.max_states,
        )
        individual_streams = []
        for i in range(B):
            individual_streams.append(k2.RnntDecodingStream(self.decoding_graph))
        decoding_streams = k2.RnntDecodingStreams(individual_streams, config)
      
        for t in range(T):
            shape, contexts = decoding_streams.get_contexts()
            contexts = contexts.to(torch.int64)
      
            decoder_out = self.forward_decoder(contexts)
      
            cur_encoder_out = torch.index_select(
                encoder_out[:, t:t + 1, :], 0, shape.row_ids(1).to(torch.int64)
            )
      
            logits = self.forward_joiner(cur_encoder_out.squeeze(1),
                decoder_out)
      
            logits = logits.squeeze(1).squeeze(1).float()
            log_probs = (logits / self.temperature).log_softmax(dim=-1)
            decoding_streams.advance(log_probs)
        decoding_streams.terminate_and_flush_to_streams()
        lattice = decoding_streams.format_output(encoder_out_lens.tolist())
      
        best_path = k2.shortest_path(lattice, use_double_scores=True)
        hyps_list = self.get_texts(best_path)
    
        return hyps_list

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
            ans = self.greedy_search(encoder_out, encoder_out_lens)
        elif self.decoding_method == 'fast_beam_search':
            ans = self.fast_beam_search(encoder_out, encoder_out_lens)
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
