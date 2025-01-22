import kaldifeat
import torch

from .torch_version import sherpa_torch_version

if torch.__version__.split("+")[0] != sherpa_torch_version.split("+")[0]:
    raise ImportError(
        f"sherpa was built using PyTorch {sherpa_torch_version}\n"
        f"But you are using PyTorch {torch.__version__} to run it"
    )

from _sherpa import (
    EndpointConfig,
    EndpointRule,
    FastBeamSearchConfig,
    FeatureConfig,
    LinearResample,
    OfflineCtcDecoderConfig,
    OfflineModelConfig,
    OfflineRecognizer,
    OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig,
    OfflineStream,
    OfflineWhisperModelConfig,
    OnlineRecognitionResult,
    OnlineRecognizer,
    OnlineRecognizerConfig,
    OnlineStream,
    SileroVadModelConfig,
    SpeakerEmbeddingExtractor,
    SpeakerEmbeddingExtractorConfig,
    VadModelConfig,
    VoiceActivityDetector,
    VoiceActivityDetectorConfig,
    cxx_flags,
)

from .http_server import HttpServer
from .utils import encode_contexts, setup_logger, str2bool
