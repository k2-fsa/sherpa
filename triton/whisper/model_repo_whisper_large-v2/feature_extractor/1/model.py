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
# Reference: https://github.com/openai/whisper/blob/main/whisper/audio.py

import triton_python_backend_utils as pb_utils
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union
import os
import time

def mel_filters(device, n_mels: int = 80) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[torch.Tensor],
    filters: torch.Tensor,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    filters: torch.Tensor

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

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
        self.device = torch.device("cuda")
        self.n_mels = 80
        self.max_num_frames = 3000
        self.filters = mel_filters(self.device, n_mels=self.n_mels)

    def compute_feature(self, wav, target):
        mel = log_mel_spectrogram(wav, self.filters)
        assert mel.shape[1] <= target, f"{mel.shape[1]} > {target}, audio is too long"
        if mel.shape[1] < target:
            mel = F.pad(mel, (0, target - mel.shape[1]), mode='constant')
        mel = mel.unsqueeze(0)
        return mel

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
        responses = []
        for i, request in enumerate(requests):
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            wav = input0.as_numpy()
            assert wav.shape[0] == 1, "Only support batch size 1"
            wav = torch.from_numpy(wav[0]).to(self.device)
            mel = self.compute_feature(wav, self.max_num_frames)
            mel = mel.cpu().numpy()
            out0 = pb_utils.Tensor("mel", mel)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)
        return responses
