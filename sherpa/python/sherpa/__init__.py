import torch

from .torch_version import sherpa_torch_version

if torch.__version__.split("+")[0] != sherpa_torch_version.split("+")[0]:
    raise ImportError(
        f"sherpa was built using PyTorch {sherpa_torch_version}\n"
        f"But you are using PyTorch {torch.__version__} to run it"
    )

from _sherpa import (
    Hypotheses,
    Hypothesis,
    RnntConformerModel,
    RnntConvEmformerModel,
    RnntEmformerModel,
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
from .lexicon import Lexicon
from .nbest import Nbest
from .utils import add_beam_search_arguments
