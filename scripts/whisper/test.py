#!/usr/bin/env python3

import torch
import whisper
import soundfile as sf

import kaldi_native_fbank as knf
from typing import Tuple
import numpy as np


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(filename: str, dim: int = 80) -> torch.Tensor:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:
        import librosa

        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = dim
    online_whisper_fbank = knf.OnlineWhisperFbank(opts)
    online_whisper_fbank.accept_waveform(16000, wave)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # mel (T, 80)

    target = 3000
    if mel.shape[0] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[: target - 50]
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)

    if mel.shape[0] < target:
        mel = torch.nn.functional.pad(
            mel, (0, 0, 0, target - mel.shape[0]), "constant", 0
        )
    else:
        mel = mel[:target]

    mel = mel.t().unsqueeze(0)

    # mel: (1, 80, 3000)

    return mel


@torch.inference_mode()
def main():
    model = whisper.load_model("tiny.en")
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )
    print(tokenizer.sot, tokenizer.language, tokenizer.task, tokenizer.sot_sequence)

    mel = compute_features("./0.wav", dim=model.dims.n_mels)

    m = torch.jit.load("model.pt")
    m.eval()
    n_layer_cross_k, n_layer_cross_v = m.run_encoder(mel)
    sot_sequence = list(tokenizer.sot_sequence)
    lang2id = dict()
    for i, n in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes):
        lang2id[n] = i
    #  sot_sequence.append(lang2id["en"])
    sot_sequence.append(tokenizer.no_timestamps)

    tokens = torch.tensor(sot_sequence).unsqueeze(0)

    n_audio = 1
    n_layer_self_k_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    n_layer_self_v_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    offset = torch.zeros(1, dtype=torch.int32).to(mel.device)

    n_layer_cross_k, n_layer_cross_v = m.run_encoder(mel)
    print(n_layer_cross_k.shape, n_layer_cross_v.shape)

    logits, n_layer_self_k_cache, n_layer_self_v_cache = m.run_decoder(
        tokens,
        n_layer_self_k_cache=n_layer_self_k_cache,
        n_layer_self_v_cache=n_layer_self_v_cache,
        n_layer_cross_k=n_layer_cross_k,
        n_layer_cross_v=n_layer_cross_v,
        offset=offset,
    )
    print(logits.shape, n_layer_self_v_cache.shape, n_layer_self_v_cache.shape)

    offset += tokens.shape[1]
    # logits.shape (batch_size, tokens.shape[1], vocab_size)
    logits = logits[0, -1]
    #  logits = logits.softmax(dim=-1)
    # for greedy search, we don't need to compute softmax or log_softmax
    max_token_id = logits.argmax(dim=-1)
    results = []

    for i in range(model.dims.n_text_ctx):
        if max_token_id == tokenizer.eot:
            break
        results.append(max_token_id.item())
        tokens = torch.tensor([[results[-1]]])

        logits, n_layer_self_k_cache, n_layer_self_v_cache = m.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        offset += 1
        logits = logits[0, -1]
        max_token_id = logits.argmax(dim=-1)
    print(results)
    print(tokenizer.decode(results))


if __name__ == "__main__":
    main()
