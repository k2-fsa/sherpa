# -*- coding: utf-8 -*- 
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import transformers
from transformers import AutoTokenizer
from typing import Dict
from pathlib import Path
import traceback

DEFAULT_SPEECH_TOKEN = "<speech>"
TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

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
        self.device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        tokenizer.padding_side = "left"
        special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer = tokenizer

        self.eos = self.tokenizer.eos_token_id
        self.default_speech_token_id = tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN
        )
        # https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/blob/main/config.json#L26
        self.vocab_size = 151936 
        self.logger = pb_utils.Logger

        # TODO: get the decoupled flag from the model config
        self.decoupled = False

    def _tokenize(self, num_speech_tokens, prompt=None):
        def preprocess(
            messages,
            tokenizer: transformers.PreTrainedTokenizer,
            max_len: int = 128,
        ) -> Dict:
            """Preprocesses the data for supervised fine-tuning."""
            texts = []
            for i, msg in enumerate(messages):
                texts.append(
                    tokenizer.apply_chat_template(
                        msg,
                        tokenize=True,
                        add_generation_prompt=False,
                        chat_template=TEMPLATE,
                        padding="longest",
                        max_length=max_len,
                        truncation=True,
                    )
                )
            max_len_texts = max([len(text) for text in texts])
            if tokenizer.padding_side == "right":
                texts = [
                    text + [tokenizer.pad_token_id] * (max_len_texts - len(text))
                    for text in texts
                ]
            else:
                texts = [
                    [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
                    for text in texts
                ]

            input_ids = torch.tensor(texts, dtype=torch.int)

            attention_mask = input_ids.ne(tokenizer.pad_token_id)

            return input_ids, attention_mask

        if prompt is None:
            prompts = [
                [
                    {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"},
                    {"role": "assistant", "content": ""},
                ]
            ]
        input_ids, _ = preprocess(prompts, self.tokenizer)
        input_ids = input_ids.tolist()[0]
        speech_token_index = input_ids.index(self.default_speech_token_id)
        prompt_ids = input_ids[:speech_token_index] + list(range(self.vocab_size, self.vocab_size + num_speech_tokens)) + input_ids[speech_token_index + 1:]
        return prompt_ids

    def _prepare_inputs(self, request, speech_embeddings, input_ids):
        """
        Prepares inputs for the language model based on the parameters in the
        request, image features, and prompt. It tokenizes prompt,
        extracts and processes additional parameters from the request:
            - max_tokens: Maximum number of tokens to generate (default: 50)
            - temperature: Controls randomness in generation (default: 0.5)
            - top_k: Top K sampling parameter (default: 1)
            - frequency_penalty: Penalizes frequent tokens (default: 0.7)
            - seed: Random seed for generation (default: 10)

        Final llm input dictionary is combined out of all processed parameters,
        prompt's tokens and image features. The latter will be passed to llm
        through `prompt_embedding_table`.

        Parameters
        ----------
        - request: The original request object containing additional parameters.
        - image_features (list): A list containing image feature tensors.
        - prompt (str): The text prompt to be processed.

        Returns
        -------
        - dict: A dictionary containing all the prepared inputs for the language model.
        """
        input_ids = np.array(input_ids, dtype=np.int32)
        # TODO: max_tokens should be in the model config
        max_tokens = 200
        input_len = input_ids.shape[0]

        embedding_args = {
            "prompt_vocab_size": np.array(
                [[speech_embeddings.shape[1]]], dtype=np.int32
            ),
            "prompt_embedding_table": speech_embeddings.detach().cpu().numpy(),
        }  

        input_dict =  {
            "input_ids": np.expand_dims(input_ids, 0),
            "input_lengths": np.array([[input_len]], dtype=np.int32),
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "runtime_top_k": np.array([[1]], dtype=np.int32),
            "end_id": np.array([[self.tokenizer.eos_token_id]], dtype=np.int32),
            "pad_id": np.array([[self.tokenizer.pad_token_id]], dtype=np.int32),
            "streaming": np.array([[0]], dtype=np.bool_),
            **embedding_args,
        }

        input_tensor_list = [pb_utils.Tensor(k, v) for k, v in input_dict.items()]
        return input_tensor_list

    def _prepare_llm_response(self, llm_request_inputs):
        """
        Prepares the response from the language model based on the provided
        inputs. Creates a `pb_utils.InferenceRequest` object with passed
        `llm_request_inputs` to send to a decoupled TensorRTLLM model.
        For each response from the language model:
            - Checks for errors and raise an exception if any are found.
            - Extracts the "output_ids" tensor from the response.
            - Determines the finish reason based on the presence of the
              end-of-sequence token or reaching the maximum length.
            - Appends the generated token IDs to `output_ids`.
            - If the finish reason is determined, decodes the output IDs to text
              and prepares the final response.

        The final response includes the generated text, finish reason,
        completion tokens, prompt tokens, and total tokens.

        Parameters
        ----------
        - llm_request_inputs (dict): A dictionary containing the inputs for the language model.

        Returns
        -------
        - pb_utils.InferenceResponse: The response object containing the generated text and additional metadata.
        """

        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=llm_request_inputs,
        )
        output_ids, output_len = [], 0
        responses = llm_request.exec(decoupled=False)
        responses = [responses]
        for llm_response in responses:
            if llm_response.has_error():
                raise pb_utils.TritonModelException(llm_response.error().message())
            stream_output_ids = (
                pb_utils.get_output_tensor_by_name(llm_response, "output_ids")
                .as_numpy()
                .flatten()
                .tolist()
            )
            # TODO: support finish_reason
            finish_reason = "test"
            if len(stream_output_ids) == 0 or (
                len(stream_output_ids) != 0
                and stream_output_ids[-1] == self.eos
            ):
                finish_reason = "stop"

            output_ids += stream_output_ids

            last_response = finish_reason != ""
            output_len = len(output_ids)
            if last_response:
                output_text = self.tokenizer.decode(output_ids).strip()
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("TRANSCRIPTS", np.array([output_text], np.object_)),
                    ]
                )
                yield response

    def _extract_speech_embeddings(self, wav, wav_len):
        wav = torch.from_numpy(wav[0]).to(self.device)
        wav_tensor = pb_utils.Tensor.from_dlpack("WAV", to_dlpack(wav.unsqueeze(0)))
        wav_len_tensor = pb_utils.Tensor("WAV_LENS", np.array([[wav_len]], np.int32))

        infer_request = pb_utils.InferenceRequest(
            model_name="speech_encoder",
            requested_output_names=["speech_features"],
            inputs=[wav_tensor, wav_len_tensor],
        )
        inference_response = infer_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            speech_features = pb_utils.get_output_tensor_by_name(inference_response, "speech_features")
            speech_features = torch.utils.dlpack.from_dlpack(speech_features.to_dlpack())

            return speech_features

    def execute(self, requests):
        responses = []
        for request in requests:
            wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
            assert wav.shape[0] == 1, "Only support batch size 1 for now"
            wav_len = pb_utils.get_input_tensor_by_name(request, "WAV_LENS").as_numpy()
            wav_len = wav_len.item()

            speech_embeddings = self._extract_speech_embeddings(wav, wav_len)
            #TODO: get the prompts from input tensors
            input_ids = self._tokenize(num_speech_tokens=speech_embeddings.shape[1])

            if self.decoupled:
                response_sender = request.get_response_sender()
            try:

                llm_request_inputs = self._prepare_inputs(
                    request, speech_embeddings, input_ids
                )
                if isinstance(llm_request_inputs, pb_utils.TritonError):
                    error = pb_utils.InferenceResponse(error=llm_request_inputs)
                    if self.decoupled:
                        response_sender.send(
                            error, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        )
                    else:
                        responses.append(error)
                llm_responses = self._prepare_llm_response(llm_request_inputs)
                
                for triton_response in llm_responses:
                    if self.decoupled:
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

                if self.decoupled:
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

            except Exception:
                self.logger.log_error(traceback.format_exc())
                # If encountering an error, send a response with err msg
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(traceback.format_exc()))

                if self.decoupled:
                    response_sender.send(error_response)
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                else:
                    responses.append(error_response)

        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses
