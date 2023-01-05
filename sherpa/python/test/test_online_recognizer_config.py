#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_online_recognizer_config_py

import unittest

import sherpa


class TestOnlineRecognizerConfig(unittest.TestCase):
    def test_constructor(self):
        config = sherpa.OnlineRecognizerConfig(nn_model="a.pt", tokens="b.txt")
        print()
        print(config)


if __name__ == "__main__":
    unittest.main()

