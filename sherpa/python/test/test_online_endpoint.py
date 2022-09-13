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
#  ctest --verbose -R  test_online_endpoint_py

import argparse
import sys
import unittest

import sherpa


class TestOnlineEndpoint(unittest.TestCase):
    def test_rule1(self):
        sys.argv = [
            "--endpoint.rule1.must-contain-nonsilence=false",
            "--endpoint.rule1.min-trailing-silence=1.0",
            "--endpoint.rule1.min-utterance-length=0.0",
        ]
        online_endpoint_parser = sherpa.add_online_endpoint_arguments()
        parser = argparse.ArgumentParser(
            parents=[online_endpoint_parser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        _, args = (
            parser.parse_args(),
            online_endpoint_parser.parse_known_args()[0],
        )
        args = vars(args)
        config = sherpa.OnlineEndpointConfig.from_args(args)

        # decoded nothing, 5 seconds of trailing silence
        t = sherpa.endpoint_detected(
            config,
            num_frames_decoded=500,
            trailing_silence_frames=500,
            frame_shift_in_seconds=0.01,
        )
        assert t is True

        # decoded something, 0.5 second of trailing silence
        f = sherpa.endpoint_detected(
            config,
            num_frames_decoded=500,
            trailing_silence_frames=50,
            frame_shift_in_seconds=0.01,
        )
        assert f is False

    def test_rule2(self):
        sys.argv = [
            "--endpoint.rule2.must-contain-nonsilence=true",
            "--endpoint.rule2.min-trailing-silence=1.0",
            "--endpoint.rule2.min-utterance-length=0.0",
        ]
        online_endpoint_parser = sherpa.add_online_endpoint_arguments()
        parser = argparse.ArgumentParser(
            parents=[online_endpoint_parser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        _, args = (
            parser.parse_args(),
            online_endpoint_parser.parse_known_args()[0],
        )
        args = vars(args)
        config = sherpa.OnlineEndpointConfig.from_args(args)

        # decoded nothing, 3 seconds of trailing silence
        r = sherpa.endpoint_detected(
            config,
            num_frames_decoded=300,
            trailing_silence_frames=300,
            frame_shift_in_seconds=0.01,
        )
        assert r is False

        # decoded something, 0.5 second of trailing silence
        s = sherpa.endpoint_detected(
            config,
            num_frames_decoded=500,
            trailing_silence_frames=50,
            frame_shift_in_seconds=0.01,
        )
        assert s is False

        # decoded something, 1.01 seconds of trailing silence
        t = sherpa.endpoint_detected(
            config,
            num_frames_decoded=500,
            trailing_silence_frames=101,
            frame_shift_in_seconds=0.01,
        )
        assert t is True

    def test_rule3(self):
        sys.argv = [
            "--endpoint.rule3.must-contain-nonsilence=false",
            "--endpoint.rule3.min-trailing-silence=0.0",
            "--endpoint.rule3.min-utterance-length=13.0",
        ]
        online_endpoint_parser = sherpa.add_online_endpoint_arguments()
        parser = argparse.ArgumentParser(
            parents=[online_endpoint_parser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        _, args = (
            parser.parse_args(),
            online_endpoint_parser.parse_known_args()[0],
        )
        args = vars(args)
        config = sherpa.OnlineEndpointConfig.from_args(args)

        # decoded nothing, 0.1 second of trailing silence
        r = sherpa.endpoint_detected(
            config,
            num_frames_decoded=1200,
            trailing_silence_frames=10,
            frame_shift_in_seconds=0.01,
        )
        assert r is False

        # decoded something, 0.1 second of trailing silence
        s = sherpa.endpoint_detected(
            config,
            num_frames_decoded=1300,
            trailing_silence_frames=10,
            frame_shift_in_seconds=0.01,
        )
        assert s is False

        # decoded something, 0.1 seconds of trailing silence
        t = sherpa.endpoint_detected(
            config,
            num_frames_decoded=1301,
            trailing_silence_frames=10,
            frame_shift_in_seconds=0.01,
        )
        assert t is True


if __name__ == "__main__":
    unittest.main()
