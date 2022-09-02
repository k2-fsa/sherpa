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
#  ctest --verbose -R  test_utils.py

import unittest

import k2

import sherpa


class TestUtils(unittest.TestCase):
    def test_count_number_trailing_zeros(self):
        assert sherpa.count_num_trailing_zeros([1, 2, 3]) == 0
        assert sherpa.count_num_trailing_zeros([1, 0, 3]) == 0

        assert sherpa.count_num_trailing_zeros([1, 0, 0]) == 2
        assert sherpa.count_num_trailing_zeros([0, 0, 0]) == 3

    def test_get_texts_and_num_trailing_blanks_case1(self):
        s1 = """
          0 1 0 0 0.0
          1 2 1 1 0.2
          2 3 1 5 0.2
          3 4 -1 -1 0.0
          4
        """

        s2 = """
          0 1 0 0 0.0
          1 2 0 1 0.2
          2 3 -1 -1 0.0
          3
        """

        s3 = """
          0 1 1 0 0.0
          1 2 0 1 0.2
          2 3 -1 -1 0.0
          3
        """

        fsa1 = k2.Fsa.from_str(s1, acceptor=False)
        fsa2 = k2.Fsa.from_str(s2, acceptor=False)
        fsa3 = k2.Fsa.from_str(s3, acceptor=False)

        fsa = k2.Fsa.from_fsas([fsa1, fsa2, fsa3])
        (
            aux_labels,
            num_trailing_blanks,
        ) = sherpa.get_texts_and_num_trailing_blanks(fsa)

        assert aux_labels == [[1, 5], [1], [1]]
        assert num_trailing_blanks == [0, 2, 1]


if __name__ == "__main__":
    unittest.main()
