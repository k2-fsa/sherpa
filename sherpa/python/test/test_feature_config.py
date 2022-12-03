#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_feature_config_py

import unittest

import torch

import sherpa


class TestFeatureConfig(unittest.TestCase):
    def test_default_constructor(self):
        config = sherpa.FeatureConfig()
        print()
        print(config)
        assert config.normalize_samples is True

    def test_constructor(self):
        config = sherpa.FeatureConfig(normalize_samples=False)
        assert config.normalize_samples is False

        config.fbank_opts.mel_opts.num_bins = 80
        config.fbank_opts.device = "cuda:1"

        assert config.fbank_opts.mel_opts.num_bins == 80
        assert config.fbank_opts.device == torch.device("cuda", 1)

        print()
        print(config)


if __name__ == "__main__":
    unittest.main()
