from dataclasses import dataclass
from typing import List, overload

import kaldifeat
import torch

class EndpointRule:
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self,
        must_contain_nonsilence=True,
        min_trailing_silence=2.0,
        min_utterance_length=0.0,
    ): ...

    must_contain_nonsilence: bool
    min_trailing_silence: float
    min_utterance_length: float

class EndpointConfig:
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self, rule1=EndpointRule(), rule2=EndpointRule(), rule3=EndpointRule()
    ): ...

    rule1: EndpointRule
    rule2: EndpointRule
    rule3: EndpointRule

class FastBeamSearchConfig:
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self,
        lg="",
        ngram_lm_scale=0.01,
        beam=20.0,
        max_states=64,
        max_contexts=8,
        allow_partial=False,
    ): ...

    lg: str
    ngram_lm_scale: float
    beam: float
    max_states: int
    max_contexts: int
    allow_partial: bool

@dataclass
class FeatureConfig:
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self,
        fbank_opts=kaldifeat.FbankOptions(),
        normalize_samples=True,
        return_waveform=False,
        nemo_normalize="",
    ): ...

    fbank_opts: kaldifeat.FbankOptions
    normalize_samples: bool
    return_waveform: bool
    nemo_normalize: str

class Hypothesis:
    timestamps: List
    num_trailing_blanks: int

    @property
    def key(self) -> str: ...
    @property
    def log_prob(self) -> float: ...
    @property
    def ys(self) -> List[int]: ...

class Hypotheses:
    def get_most_probable(self, length_norm: bool) -> Hypothesis: ...

class LinearResample:
    def reset(self) -> None: ...
    def resample(self) -> torch.Tensor: ...

    input_sample_rate: int
    output_sample_rate: int

class OfflineCtcDecoderConfig:
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self,
        modified=True,
        hlg="",
        search_beam=20,
        output_beam=8,
        min_active_states=30,
        max_active_states=10000,
        lm_scale=1.0,
    ): ...

    modified: bool
    hlg: str
    search_beam: float
    output_beam: float
    min_active_states: int
    max_active_states: int
    lm_scale: float

    def validate(self) -> None: ...

class OfflineRecognizerConfig:
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self,
        nn_model,
        tokens,
        use_gpu=False,
        decoding_method="greedy_search",
        num_active_paths=4,
        context_score=1.5,
        ctc_decoder_config=OfflineCtcDecoderConfig(),
        feat_config=FeatureConfig(),
        fast_beam_search_config=FastBeamSearchConfig(),
    ): ...

    ctc_decoder_config: OfflineCtcDecoderConfig
    feat_config: FeatureConfig
    fast_beam_search_config: FastBeamSearchConfig
    nn_model: str
    tokens: str
    use_gpu: bool
    decoding_method: str
    num_active_paths: int
    context_score: float

    def validate(self) -> None: ...

class OfflineRecognitionResult:
    text: str
    tokens: List[str]
    timestamps: List[float]

    def as_json_string(self) -> str: ...

class OfflineStream:
    def accept_wave_file(self, filename: str) -> None: ...
    @overload
    def accept_samples(self, samples: List[float]) -> None: ...
    @overload
    def accept_samples(self, samples: torch.Tensor) -> None: ...
    @property
    def result(self) -> OfflineRecognitionResult: ...
    accept_waveform = accept_samples

class OfflineRecognizer:
    def __init__(self, config: OfflineRecognizerConfig) -> None: ...
    @overload
    def create_stream(self) -> OfflineStream: ...
    @overload
    def create_stream(
        self, contexts_list: List[List[int]]
    ) -> OfflineStream: ...
    def decode_stream(self, s: OfflineStream) -> None: ...
    def decode_streams(self, ss: List[OfflineStream]) -> None: ...

class OnlineRecognizerConfig:
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self,
        nn_model,
        tokens,
        encoder_model="",
        decoder_model="",
        joiner_model="",
        use_gpu=False,
        use_endpoint=False,
        decoding_method="greedy_search",
        num_active_paths=4,
        left_context=64,
        right_context=0,
        chunk_size=12,
        feat_config=FeatureConfig(),
        endpoint_config=EndpointConfig(),
        fast_beam_search_config=FastBeamSearchConfig(),
    ): ...
    feat_config: FeatureConfig
    endpoint_config: EndpointConfig
    fast_beam_search_config: FastBeamSearchConfig
    nn_model: str
    tokens: str
    encoder_model: str
    decoder_model: str
    joiner_model: str
    use_gpu: bool
    use_endpoint: bool
    decoding_method: str
    num_active_paths: int
    left_context: int
    right_context: int
    chunk_size: int

    def validate(self) -> None: ...

class OnlineRecognitionResult:
    @property
    def text(self) -> str: ...
    @property
    def tokens(self) -> List[str]: ...
    @property
    def timestamps(self) -> float: ...
    @property
    def segment(self) -> int: ...
    @property
    def start_time(self) -> float: ...
    @property
    def is_final(self) -> bool: ...
    def as_json_string(self) -> str: ...

class OnlineStream:
    def accept_waveform(
        self, sampling_rate: int, waveform: torch.Tensor
    ) -> None: ...
    def input_finished(self) -> None: ...

class OnlineRecognizer:
    def __init__(self, config: OnlineRecognizerConfig): ...
    def create_stream(self) -> OnlineStream: ...
    def is_ready(self, s: OnlineStream) -> bool: ...
    def is_endpoint(self, s: OnlineStream) -> bool: ...
    def decode_stream(self, s: OnlineStream) -> bool: ...
    def decode_streams(self, ss: List[OnlineStream]) -> None: ...
    def get_result(self, s: OnlineStream) -> OnlineRecognitionResult: ...
    @property
    def config(self) -> OnlineRecognizerConfig: ...
