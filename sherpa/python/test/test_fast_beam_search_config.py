#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_fast_beam_search_config_py

import unittest

import sherpa


class TestFastBeamSearchConfig(unittest.TestCase):
    def test_default_constructor(self):
        config = sherpa.FastBeamSearchConfig()
        print()
        print(config)
        assert config.lg == ""
        assert abs(config.ngram_lm_scale - 0.01) < 1e-5, config.ngram_lm_scale
        assert config.beam == 20.0
        assert config.max_states == 64
        assert config.max_contexts == 8
        assert config.allow_partial is False

    def test_constructor(self):
        config = sherpa.FastBeamSearchConfig(lg="a.pt", allow_partial=True)
        assert config.lg == "a.pt"
        assert config.allow_partial is True

        print()
        print(config)


if __name__ == "__main__":
    unittest.main()
