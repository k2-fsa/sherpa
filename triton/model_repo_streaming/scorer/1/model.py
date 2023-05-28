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
import sentencepiece as spm
from typing import Union, List

from icefall.lexicon import Lexicon
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

        # Get INPUT configuration
        if "GPU" in args['model_instance_kind']:
            self.device = f"cuda:{args['model_instance_device_id']}"
        else:
            self.device = "cpu"

        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        self.encoder_dim = encoder_config['dims'][-1]
        
        
        self.init_sentence_piece(self.model_config['parameters'])

        self.decoding_method = self.model_config['parameters']['decoding_method']
        # parameters for fast beam search
        if self.decoding_method == 'fast_beam_search':
            self.temperature = float(self.model_config['parameters']['temperature'])

            self.beam = int(self.model_config['parameters']['beam'])
            self.max_contexts = int(self.model_config['parameters']['max_contexts'])
            self.max_states = int(self.model_config['parameters']['max_states'])
            
            self.fast_beam_config = k2.RnntDecodingConfig(
                vocab_size=self.vocab_size,
                decoder_history_len=self.context_size,
                beam=self.beam,
                max_contexts=self.max_contexts,
                max_states=self.max_states,
            )

            self.decoding_graph = k2.trivial_graph(
                    self.vocab_size - 1, device=self.device
            )
        # use to record every sequence state
        self.seq_states = {}
        print("Finish Init")

    def init_sentence_piece(self, parameters):
        for key,value in parameters.items():
            parameters[key] = value["string_value"]
        self.context_size = int(parameters['context_size'])
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


    def forward_decoder(self,contexts):
        decoder_input = np.asarray(contexts, dtype=np.int64)
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

    def greedy_search(self, encoder_out, encoder_out_lens, hyps_list):
        emitted = False
        # add blank_id as prefix
        hyps_list = [[self.blank_id] * self.context_size + h for h in hyps_list]
        contexts = [h[-self.context_size:] for h in hyps_list]
        decoder_out = self.forward_decoder(contexts)
        assert encoder_out.shape[0] == decoder_out.shape[0]
        for t in range(encoder_out.shape[1]):
            if emitted:
                contexts = [h[-self.context_size:] for h in hyps_list]
                decoder_out = self.forward_decoder(contexts)

            cur_encoder_out = encoder_out[:,t]
            logits = self.forward_joiner(cur_encoder_out, decoder_out)

            assert logits.ndim == 2, logits.shape
            y = logits.argmax(dim=1).tolist()
            
            for i, v in enumerate(y):
                if v not in (self.blank_id, self.unk_id):
                    hyps_list[i].append(v)
                    emitted = True
        # remove prefix blank_id
        hyps_list = [h[self.context_size:] for h in hyps_list]
        return hyps_list

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

    def fast_beam_search(self, encoder_out, encoder_out_lens, states_list):
        streams_list = [state[0] for state in states_list]
        processed_lens_list = [state[1] for state in states_list]

        decoding_streams = k2.RnntDecodingStreams(streams_list, self.fast_beam_config)

        for t in range(encoder_out.shape[1]):
            shape, contexts = decoding_streams.get_contexts()
            contexts = contexts.to(torch.int64)

            decoder_out = self.forward_decoder(contexts)

            cur_encoder_out = torch.index_select(
                encoder_out[:, t, :], 0, shape.row_ids(1).to(torch.int64)
            )

            logits = self.forward_joiner(cur_encoder_out,
                decoder_out)

            logits = logits.squeeze(1).squeeze(1).float()
            log_probs = (logits / self.temperature).log_softmax(dim=-1)
            decoding_streams.advance(log_probs)
        decoding_streams.terminate_and_flush_to_streams()
        lattice = decoding_streams.format_output(processed_lens_list)

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
        batch_idx = 0
        encoder_max_len = 0

        batch_idx2_corrid = {}
    
        states_list = []
        end_idx = set()

        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            in_1 = pb_utils.get_input_tensor_by_name(request, "encoder_out_lens")

            # TODO: directly use torch tensor from_dlpack(in_0.to_dlpack())
            batch_encoder_out_list.append(in_0.as_numpy())
            # For streaming ASR, assert each request sent from client has batch size 1.
            assert batch_encoder_out_list[-1].shape[0] == 1            
            encoder_max_len = max(encoder_max_len, batch_encoder_out_list[-1].shape[1])

            cur_b_lens = in_1.as_numpy()
            batch_encoder_lens_list.append(cur_b_lens)

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
                if self.decoding_method == 'fast_beam_search':
                    processed_len = cur_b_lens
                    state = [k2.RnntDecodingStream(self.decoding_graph), processed_len]
                else:
                    state = []
                self.seq_states[corrid] = state

            if end and ready:
                end_idx.add(batch_idx)
    
            if ready:
                state = self.seq_states[corrid]
                batch_idx2_corrid[batch_idx] = corrid
                states_list.append(state)

            batch_idx += 1

        encoder_out_array = np.zeros((batch_idx, encoder_max_len, self.encoder_dim),
                                  dtype=self.data_type)
        encoder_out_lens_array = np.zeros(batch_idx, dtype=np.int32)

        for i, t in enumerate(batch_encoder_out_list):
            encoder_out_array[i, 0:t.shape[1]] = t
            encoder_out_lens_array[i] = batch_encoder_lens_list[i]
    
        encoder_out = torch.from_numpy(encoder_out_array)
        encoder_out_lens = torch.from_numpy(encoder_out_lens_array)

        if self.decoding_method == "fast_beam_search":
            hyps_list = self.fast_beam_search(encoder_out, encoder_out_lens, states_list)
        else:
            hyps_list = self.greedy_search(encoder_out, encoder_out_lens, states_list)

        responses = []
        for i in range(len(hyps_list)):
            hyp = hyps_list[i]
            if hasattr(self.tokenizer, 'token_table'):
                sent = [self.tokenizer.token_table[idx] for idx in hyp]
            else:
                sent = self.tokenizer.decode(hyp).split()
            sent = np.array(sent)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", sent.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
            # update states
            corr = batch_idx2_corrid[i]
            if i in end_idx:
                del self.seq_states[corr]
            else:
                if self.decoding_method == 'fast_beam_search':
                    self.seq_states[corr][1] += batch_encoder_lens_list[i]  # stream decoding state is updated in fast_beam_search
                else:
                    self.seq_states[corr] = hyp
                    
        assert len(requests) == len(responses)
        return responses
    
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')