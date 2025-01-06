#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import numpy as np
from funasr_torch import SenseVoiceSmall


def generate_tokens(m):
    sp = m.tokenizer.sp
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")


def generate_bpe_model(m):
    with open("bpe.model", "wb") as f:
        f.write(m.tokenizer.sp.serialized_model_proto())


def main():
    model_dir = "iic/SenseVoiceSmall"
    model = SenseVoiceSmall(model_dir, batch_size=1, device="cpu")

    generate_tokens(model)
    generate_bpe_model(model)

    meta_data = {
        "model_type": "SenseVoiceSmall",
        "lfr_window_size": str(model.frontend.lfr_m),
        "lfr_window_shift": str(model.frontend.lfr_n),
        "neg_mean": model.frontend.cmvn[0].astype(np.float32).tobytes(),
        "inv_stddev": model.frontend.cmvn[1].astype(np.float32).tobytes(),
        "vocab_size": str(model.tokenizer.get_vocab_size()),
        "normalize_samples": "0",  # input should be in the range [-32768, 32767]
        "version": "1",
        "model_author": "iic",
        "maintainer": "k2-fsa",
        "lang_auto": str(model.lid_dict["auto"]),
        "lang_zh": str(model.lid_dict["zh"]),
        "lang_en": str(model.lid_dict["en"]),
        "lang_yue": str(model.lid_dict["yue"]),  # cantonese
        "lang_ja": str(model.lid_dict["ja"]),
        "lang_ko": str(model.lid_dict["ko"]),
        "lang_nospeech": str(model.lid_dict["nospeech"]),
        "with_itn": str(model.textnorm_dict["withitn"]),
        "without_itn": str(model.textnorm_dict["woitn"]),
        "url": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall",
    }
    print(meta_data)
    model.ort_infer.save("model.pt", _extra_files=meta_data)


if __name__ == "__main__":
    main()
