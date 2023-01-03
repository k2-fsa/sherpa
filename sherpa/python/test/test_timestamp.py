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
#  ctest --verbose -R  test_timestamp_py

import unittest

import sherpa


class TestTimeStamp(unittest.TestCase):
    def test_convert_timestamp(self):
        subsampling_factor = 4
        frame_shift_ms = 10

        frames = [0, 1, 2, 3, 5, 8, 10]
        timestamps = sherpa.convert_timestamp(
            frames,
            subsampling_factor=4,
            frame_shift_ms=10,
        )
        for i in range(len(frames)):
            assert timestamps[i] == (
                frames[i] * subsampling_factor * frame_shift_ms / 1000
            ), (frames[i], timestamps[i])


if __name__ == "__main__":
    unittest.main()
