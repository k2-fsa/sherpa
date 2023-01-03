#!/usr/bin/env python3
# noqa
# To run this single test, use
#
#  ctest --verbose -R  test_endpoint_rule_py

import unittest
from pathlib import Path

import sherpa


class TestEndpointRule(unittest.TestCase):
    def test_default_constructor(self):
        rule = sherpa.EndpointRule()
        print()
        print(rule)
        assert rule.must_contain_nonsilence is True
        assert rule.min_trailing_silence == 2.0, rule.min_trailing_silence
        assert rule.min_utterance_length == 0.0, rule.min_utterance_length

    def test_constructor(self):
        rule = sherpa.EndpointRule(
            must_contain_nonsilence=False,
            min_trailing_silence=10,
            min_utterance_length=20,
        )
        print()
        print(rule)
        assert rule.must_contain_nonsilence is False
        assert rule.min_trailing_silence == 10, rule.min_trailing_silence
        assert rule.min_utterance_length == 20, rule.min_utterance_length


if __name__ == "__main__":
    unittest.main()
