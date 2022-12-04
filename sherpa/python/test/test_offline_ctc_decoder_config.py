#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_offline_ctc_decoder_config_py

import unittest

import sherpa


class TestOfflineCtcDecoderConfig(unittest.TestCase):
    def test_default_constructor(self):
        config = sherpa.OfflineCtcDecoderConfig()
        print(config)
        assert config.modified is True
        assert config.hlg == ""
        assert config.search_beam == 20
        assert config.output_beam == 8
        assert config.min_active_states == 20
        assert config.max_active_states == 10000

    def test_constructor(self):
        config = sherpa.OfflineCtcDecoderConfig(
            modified=False,
            hlg="a.pt",
            search_beam=22,
            output_beam=10,
            min_active_states=10,
            max_active_states=300,
        )
        print(config)
        assert config.modified is False
        assert config.hlg == "a.pt"
        assert config.search_beam == 22
        assert config.output_beam == 10
        assert config.min_active_states == 10
        assert config.max_active_states == 300


if __name__ == "__main__":
    unittest.main()
