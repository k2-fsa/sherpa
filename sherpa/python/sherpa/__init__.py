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
