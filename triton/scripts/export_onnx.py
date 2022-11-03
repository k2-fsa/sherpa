#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
#           2022 Nvidia (Author: Yuekai Zhang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

# This script converts several saved checkpoints
# to a single one using model averaging.
"""

Usage:

(1) Export to ONNX format with streaming ASR model
mv export_onnx.py <your_icefall_path>/pruned_transducer_stateless3/
mv onnx_triton_utils.py <your_icefall_path>/pruned_transducer_stateless3/
./pruned_transducer_stateless3/export_onnx.py \
    --exp-dir ./icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/exp \
    --bpe-model ./icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/data/lang_bpe_500/bpe.model \
    --epoch 999 \
    --avg 1 \
    --streaming-model 1\
    --causal-convolution 1 \
    --onnx 1 \
    --fp16

(2) Export to ONNX format with offline ASR model
./pruned_transducer_stateless3/export_onnx.py \
    --exp-dir ./icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/exp \
    --bpe-model ./icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/data/lang_bpe_500/bpe.model \
    --epoch 999 \
    --avg 1 \
    --onnx 1 \
    --fp16

It will generate the following six files in the given `exp_dir`.
TODO: Check `onnx_check.py` for how to use them.

    - encoder.onnx
    - decoder.onnx
    - joiner.onnx
    - encoder_fp16.onnx
    - decoder_fp16.onnx
    - joiner_fp16.onnx

Note: If you don't want to train a model from scratch, we have
provided one for you. You can get it at

https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625

with the following commands:

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625
    # You will find the pre-trained model in icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/exp
"""

import argparse
import logging
from pathlib import Path

import onnx
import onnxruntime
import sentencepiece as spm
import torch
import torch.nn as nn

from onnx_triton_utils import StreamingEncoder, OfflineEncoder, get_transducer_model
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_params

from icefall.checkpoint import (
    average_checkpoints,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import str2bool

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="""It specifies the checkpoint to use for averaging.
        Note: Epoch counts from 0.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless3/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--onnx",
        type=str2bool,
        default=False,
        help="""If True, --jit is ignored and it exports the model
        to onnx format. Three files will be generated:

            - encoder.onnx
            - decoder.onnx
            - joiner.onnx

        Check ./onnx_check.py and ./onnx_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    parser.add_argument(
        "--streaming-model",
        type=str2bool,
        default=False,
        help="""Whether to export a streaming model, if the models in exp-dir
        are streaming model, this should be True.
        """,
    )

    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')

    add_model_arguments(parser)

    return parser

def to_numpy(tensors):
    out = []
    if type(tensors) == torch.tensor:
        tensors = [tensors]
    for tensor in tensors:
        if tensor.requires_grad:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.cpu().numpy()
        out.append(tensor)
    return out

def export_encoder_model_onnx_streaming(
    encoder_model: nn.Module,
    encoder_filename: str,
    opset_version: int = 11,
    left_context: int = 64,
    right_context: int = 4,
    chunk_size: int = 16,
    warmup: float = 1.0,
) -> None:
    """Export the given encoder model to ONNX format.
    The exported model has two inputs:

        - x, a tensor of shape (N, T, C); dtype is torch.float32
        - x_lens, a tensor of shape (N,); dtype is torch.int64

    and it has two outputs:

        - encoder_out, a tensor of shape (N, T, C)
        - encoder_out_lens, a tensor of shape (N,)

    Note: The warmup argument is fixed to 1.

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    encoder_model = StreamingEncoder(
        encoder_model, left_context, right_context, chunk_size, warmup
    )
    encoder_model.eval()
    x = torch.zeros(1, 51, 80, dtype=torch.float32)
    x_lens = torch.tensor([51], dtype=torch.int64) #TODO FIX int32
    states = [
        torch.zeros(
            1,
            encoder_model.left_context,
            encoder_model.encoder_layers,
            encoder_model.d_model,
        ),
        torch.zeros(
            1,
            encoder_model.cnn_module_kernel - 1,
            encoder_model.encoder_layers,
            encoder_model.d_model,
        ),
    ]

    attn_cache, cnn_cache = states[0], states[1]

    processed_lens = torch.tensor([0], dtype=torch.int64)

    processed_lens = processed_lens.unsqueeze(-1)

    #  encoder_model = torch.jit.script(encoder_model)
    # It throws the following error for the above statement
    #
    # RuntimeError: Exporting the operator __is_ to ONNX opset version
    # 11 is not supported. Please feel free to request support or
    # submit a pull request on PyTorch GitHub.
    #
    # I cannot find which statement causes the above error.
    # torch.onnx.export() will use torch.jit.trace() internally, which
    # works well for the current reworked model

    torch.onnx.export(
        encoder_model,
        (x, x_lens, attn_cache, cnn_cache, processed_lens),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "x",
            "x_lens",
            "attn_cache",
            "cnn_cache",
            "processed_lens",
        ],
        output_names=[
            "encoder_out",
            "encoder_out_lens",
            "next_attn_cache",
            "next_cnn_cache",
            "next_processed_lens",
        ],
        dynamic_axes={
            "x": {0: "B", 1: "T"},
            "x_lens": {0: "B"},
            "attn_cache": {0: "B"},
            "cnn_cache": {0: "B"},
            "processed_lens": {0: "B"},
            "encoder_out": {0: "B", 1: "T"},
            "encoder_out_lens": {0: "B"},
            "next_attn_cache": {0: "B"},
            "next_cnn_cache": {0: "B"},
            "next_processed_lens": {0: "B"},
        },
    )

    with torch.no_grad():
        o0, o1, o2, o3, o4 = encoder_model(x, x_lens, attn_cache, cnn_cache, processed_lens)
    print(o4.shape,o4)

    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(str(encoder_filename),
                                               providers=providers)
    ort_inputs = {'x': to_numpy(x),
                  'x_lens': to_numpy(x_lens),
                  'attn_cache': to_numpy(attn_cache),
                  'cnn_cache': to_numpy(cnn_cache),
                  'processed_lens': to_numpy(processed_lens)}
    ort_outs = ort_session.run(None, ort_inputs)

    logging.info(f"Saved to {encoder_filename}")


def export_encoder_model_onnx_triton(
    encoder_model: nn.Module,
    encoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the given encoder model to ONNX format.
    The exported model has two inputs:

        - x, a tensor of shape (N, T, C); dtype is torch.float32
        - x_lens, a tensor of shape (N,); dtype is torch.int64

    and it has two outputs:

        - encoder_out, a tensor of shape (N, T, C)
        - encoder_out_lens, a tensor of shape (N,)

    Note: The warmup argument is fixed to 1.

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    encoder_model = OfflineEncoder(encoder_model)
    encoder_model.eval()
    x = torch.zeros(1, 51, 80, dtype=torch.float32)
    x_lens = torch.tensor([51], dtype=torch.int64) #TODO FIX int32

    torch.onnx.export(
        encoder_model,
        (x, x_lens),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "speech",
            "speech_lengths",
        ],
        output_names=[
            "encoder_out",
            "encoder_out_lens",
        ],
        dynamic_axes={
            "speech": {0: "B", 1: "T"},
            "speech_lengths": {0: "B"},
            "encoder_out": {0: "B", 1: "T"},
            "encoder_out_lens": {0: "B"},
        },
    )

    logging.info(f"Saved to {encoder_filename}")

def export_decoder_model_onnx_triton(
    decoder_model: nn.Module,
    decoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the decoder model to ONNX format.

    The exported model has one input:

        - y: a torch.int64 tensor of shape (N, decoder_model.context_size)

    and has one output:

        - decoder_out: a torch.float32 tensor of shape (N, 1, C)

    Note: The argument need_pad is fixed to False.

    Args:
      decoder_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)

    decoder_model.eval()

    # Note(fangjun): torch.jit.trace() is more efficient than torch.jit.script()
    # in this case
    torch.onnx.export(
        decoder_model,
        (y,),
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y"],
        output_names=["decoder_out"],
        dynamic_axes={
            "y": {0: "N"},
            "decoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_onnx_triton(
    joiner_model: nn.Module,
    joiner_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the joiner model to ONNX format.
    The exported model has two inputs:

        - encoder_out: a tensor of shape (N, encoder_out_dim)
        - decoder_out: a tensor of shape (N, decoder_out_dim)

    and has one output:

        - joiner_out: a tensor of shape (N, vocab_size)

    Note: The argument project_input is fixed to True. A user should not
    project the encoder_out/decoder_out by himself/herself. The exported joiner
    will do that for the user.
    """
    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)

    project_input = True
    joiner_model.eval()
    # Note: It uses torch.jit.trace() internally
    torch.onnx.export(
        joiner_model,
        (encoder_out, decoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out", "decoder_out"],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    logging.info(f"Saved to {joiner_filename}")

@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    if params.streaming_model:
        assert params.causal_convolution

    logging.info(params)

    logging.info("About to create model")
    if params.onnx:
        model = get_transducer_model(params, enable_giga=False)
    else:
        raise NotImplementedError

    model.to(device)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )
    elif params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )

    model.to("cpu")
    model.eval()

    if params.onnx is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        opset_version = 11
        logging.info("Exporting to onnx format")
        encoder_filename = params.exp_dir / "encoder.onnx"
        if params.streaming_model:
            export_encoder_model_onnx_streaming(
                model.encoder, encoder_filename, opset_version=opset_version
            )
        else:
            export_encoder_model_onnx_triton(
                model.encoder, encoder_filename, opset_version=opset_version
            )

        decoder_filename = params.exp_dir / "decoder.onnx"
        # if not params.streaming_model:
        #     raise NotImplementedError
        # else:
        #      export_decoder_model_onnx_triton(
        #         model.decoder,
        #         decoder_filename,
        #         opset_version=opset_version,
        #     )   

        export_decoder_model_onnx_triton(
        model.decoder,
        decoder_filename,
        opset_version=opset_version,
        )           

        joiner_filename = params.exp_dir / "joiner.onnx"
        export_joiner_model_onnx_triton(
            model.joiner,
            joiner_filename,
            opset_version=opset_version,
        )
        # if not params.streaming_model:
        #     raise NotImplementedError
        # else:
        #     export_joiner_model_onnx_triton(
        #         model.joiner,
        #         joiner_filename,
        #         opset_version=opset_version,
        #     )
        if params.fp16:
            try:
                import onnxmltools
                from onnxmltools.utils.float16_converter import convert_float_to_float16
            except ImportError:
                print('Please install onnxmltools!')
                import sys
                sys.exit(1)
            def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
                onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
                onnx_fp16_model = convert_float_to_float16(onnx_fp32_model)
                onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)
            encoder_fp16_filename = params.exp_dir / "encoder_fp16.onnx"
            export_onnx_fp16(encoder_filename, encoder_fp16_filename)
            decoder_fp16_filename = params.exp_dir / "decoder_fp16.onnx"
            export_onnx_fp16(decoder_filename, decoder_fp16_filename)
            joiner_fp16_filename = params.exp_dir / "joiner_fp16.onnx"
            export_onnx_fp16(joiner_filename, joiner_fp16_filename)
    else:
        logging.info("Not using onnx")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
