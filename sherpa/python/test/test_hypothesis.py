#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
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

# To run this single test, use
#
#  ctest --verbose -R  test_hypothesis_py

import unittest

import sherpa


class TestHypothesis(unittest.TestCase):
    def test_hypothesis_default_constructor(self):
        hyp = sherpa.Hypothesis()
        assert hyp.ys == [], hyp.ys
        assert hyp.log_prob == 0, hyp.log_prob

    def test_hypothesis_constructor(self):
        hyp = sherpa.Hypothesis(ys=[1, 2, 3], log_prob=0.5)
        assert hyp.ys == [1, 2, 3], hyp.ys
        assert hyp.log_prob == 0.5, hyp.log_prob
        assert hyp.key == "-".join(map(str, hyp.ys)) == "1-2-3"


if __name__ == "__main__":
    unittest.main()
