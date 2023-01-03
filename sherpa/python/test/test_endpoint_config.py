#!/usr/bin/env python3
# noqa
# To run this single test, use
#
#  ctest --verbose -R  test_endpoint_config_py

import unittest
from pathlib import Path

import sherpa


class TestEndpointConfig(unittest.TestCase):
    def test_default_constructor(self):
        config = sherpa.EndpointConfig()
        print()
        print(config)

        assert config.rule1.must_contain_nonsilence is False
        assert (
            abs(config.rule1.min_trailing_silence - 2.40) < 1e-4
        ), config.rule1.min_trailing_silence
        assert config.rule1.min_utterance_length == 0

        assert config.rule2.must_contain_nonsilence is True
        assert (
            abs(config.rule2.min_trailing_silence - 1.20) < 1e-4
        ), config.rule2.min_trailing_silence
        assert config.rule2.min_utterance_length == 0

        assert config.rule3.must_contain_nonsilence is False
        assert (
            config.rule3.min_trailing_silence == 0
        ), config.rule3.min_trailing_silence
        assert config.rule3.min_utterance_length == 20

    def test_constructor(self):
        config = sherpa.EndpointConfig(
            rule1=sherpa.EndpointRule(False, 3.0, 0),
            rule2=sherpa.EndpointRule(True, 2.20, 0),
            rule3=sherpa.EndpointRule(False, 0, 10),
        )
        print()
        print(config)

        assert config.rule1.must_contain_nonsilence is False
        assert (
            abs(config.rule1.min_trailing_silence - 3.0) < 1e-4
        ), config.rule1.min_trailing_silence
        assert config.rule1.min_utterance_length == 0

        assert config.rule2.must_contain_nonsilence is True
        assert (
            abs(config.rule2.min_trailing_silence - 2.20) < 1e-4
        ), config.rule2.min_trailing_silence
        assert config.rule2.min_utterance_length == 0

        assert config.rule3.must_contain_nonsilence is False
        assert (
            config.rule3.min_trailing_silence == 0
        ), config.rule3.min_trailing_silence
        assert config.rule3.min_utterance_length == 10


if __name__ == "__main__":
    unittest.main()
