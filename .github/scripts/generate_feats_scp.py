#!/usr/bin/env python3
"""
Usage:

    ./generate_feats_scp.py scp:wav.scp ark,scp:feats.ark,feats.scp

It generates `feats.ark` and `feats.scp` from `wav.scp`.

Different from Kaldi's `compute-fbank-feats`, this scripts uses
normalized samples in the range [-1, 1] to compute features.
"""

import sys

import kaldi_native_io as kio
import kaldifeat
import torch


def main():
    rspecifier = sys.argv[1]
    wspecifier = sys.argv[2]

    opts = kaldifeat.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    fbank = kaldifeat.Fbank(opts)

    with kio.SequentialWaveReader(rspecifier) as ki:
        with kio.FloatMatrixWriter(wspecifier) as ko:
            for key, value in ki:
                tensor = torch.from_numpy(value.data.numpy()).clone().squeeze(0)
                tensor = tensor / 32768
                features = fbank(tensor)
                ko.write(key, features.numpy())


if __name__ == "__main__":
    main()
