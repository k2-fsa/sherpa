import torch

import sherpa
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_url, cached_download, hf_hub_download
from .offline_asr import OfflineAsr


def get_hfconfig(model_id, config_name="hf_config"):
    info = HfApi().model_info(repo_id=model_id)
    if config_name is not None:
        if config_name in info:
            return info[config_name]
        else:
            raise ValueError("Config section " + config_name + " not found")
    else:
        return info


def model_from_hfconfig(hf_repo, hf_config):
    nn_model_filename = hf_hub_download(hf_repo, hf_config["nn_model_filename"])
    token_filename = (
        hf_hub_download(hf_repo, hf_config["token_filename"])
        if "token_filename" in hf_config
        else None
    )
    bpe_model_filename = (
        hf_hub_download(hf_repo, hf_config["bpe_model_filename"])
        if "bpe_model_filename" in hf_config
        else None
    )
    decoding_method = hf_config["decoding_method"]
    sample_rate = hf_config["sample_rate"]
    num_active_paths = hf_config["num_active_paths"]

    assert decoding_method in ("greedy_search", "modified_beam_search"), decoding_method

    if decoding_method == "modified_beam_search":
        assert num_active_paths >= 1, num_active_paths

    if bpe_model_filename:
        assert token_filename is None

    if token_filename:
        assert bpe_model_filename is None

    return OfflineAsr(
        nn_model_filename,
        bpe_model_filename,
        token_filename,
        decoding_method,
        num_active_paths,
        sample_rate,
    )


def transcribe_batch_from_tensor(model, batch):
    return model.decode_waves([batch])[0]
