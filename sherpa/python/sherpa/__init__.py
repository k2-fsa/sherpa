import kaldifeat
import torch

from .torch_version import sherpa_torch_version

if torch.__version__.split("+")[0] != sherpa_torch_version.split("+")[0]:
    raise ImportError(
        f"sherpa was built using PyTorch {sherpa_torch_version}\n"
        f"But you are using PyTorch {torch.__version__} to run it"
    )

from _sherpa import (
    FastBeamSearchConfig,
    FeatureConfig,
    Hypotheses,
    Hypothesis,
    OfflineCtcDecoderConfig,
    RnntConformerModel,
    RnntConvEmformerModel,
    RnntEmformerModel,
    RnntLstmModel,
    cxx_flags,
    greedy_search,
    modified_beam_search,
    streaming_greedy_search,
    streaming_modified_beam_search,
)

from .decode import (
    VALID_FAST_BEAM_SEARCH_METHOD,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_one_best,
)
from .http_server import HttpServer
from .lexicon import Lexicon
from .nbest import Nbest
from .online_endpoint import (
    OnlineEndpointConfig,
    add_online_endpoint_arguments,
    endpoint_detected,
)
from .timestamp import convert_timestamp
from .utils import (
    add_beam_search_arguments,
    count_num_trailing_zeros,
    get_fast_beam_search_results,
    setup_logger,
)
