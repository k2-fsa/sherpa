"""
This file implements endpoint detection using
https://github.com/kaldi-asr/kaldi/blob/master/src/online2/online-endpoint.h
as a reference
"""
import argparse
from dataclasses import dataclass
from typing import Optional

from .utils import str2bool


@dataclass
class OnlineEndpointRule:
    # If True, for this endpointing rule to apply there must
    # be nonsilence in the best-path traceback.
    # For RNN-T decoding, a non-blank token is considered as non-silence
    must_contain_nonsilence: bool

    # This endpointing rule requires duration of trailing silence
    # (in seconds) to be >= this value.
    min_trailing_silence: float

    # This endpointing rule requires utterance-length (in seconds)
    # to be >= this value.
    min_utterance_length: float


class OnlineEndpointConfig:
    def __init__(
        self,
        rule1: Optional[OnlineEndpointRule] = None,
        rule2: Optional[OnlineEndpointRule] = None,
        rule3: Optional[OnlineEndpointRule] = None,
    ):
        # rule1 times out after 5 seconds of silence, even if we decoded nothing.
        self.rule1 = (
            rule1
            if rule1 is not None
            else OnlineEndpointRule(
                must_contain_nonsilence=False,
                min_trailing_silence=5.0,
                min_utterance_length=0.0,
            )
        )

        # rule2 times out after 2.0 seconds of silence after decoding something,
        self.rule2 = (
            rule2
            if rule2 is not None
            else OnlineEndpointRule(
                must_contain_nonsilence=True,
                min_trailing_silence=2.0,
                min_utterance_length=0.0,
            )
        )
        # rule3 times out after the utterance is 20 seconds long, regardless of
        # anything else.
        self.rule3 = (
            rule3
            if rule3 is not None
            else OnlineEndpointRule(
                must_contain_nonsilence=False,
                min_trailing_silence=0.0,
                min_utterance_length=20.0,
            )
        )

    @staticmethod
    def from_args(args: dict) -> "OnlineEndpointConfig":
        """
        Args:
          args:
            It contains the arguments parsed from
            :func:`add_online_endpoint_arguments`
        """
        rule1 = OnlineEndpointRule(
            must_contain_nonsilence=args[
                "endpoint_rule1_must_contain_nonsilence"
            ],
            min_trailing_silence=args["endpoint_rule1_min_trailing_silence"],
            min_utterance_length=args["endpoint_rule1_min_utterance_length"],
        )

        rule2 = OnlineEndpointRule(
            must_contain_nonsilence=args[
                "endpoint_rule2_must_contain_nonsilence"
            ],
            min_trailing_silence=args["endpoint_rule2_min_trailing_silence"],
            min_utterance_length=args["endpoint_rule2_min_utterance_length"],
        )

        rule3 = OnlineEndpointRule(
            must_contain_nonsilence=args[
                "endpoint_rule3_must_contain_nonsilence"
            ],
            min_trailing_silence=args["endpoint_rule3_min_trailing_silence"],
            min_utterance_length=args["endpoint_rule3_min_utterance_length"],
        )

        return OnlineEndpointConfig(rule1=rule1, rule2=rule2, rule3=rule3)


def _add_rule_arguments(
    parser: argparse.ArgumentParser,
    prefix: str,
    rule: OnlineEndpointRule,
):
    p = prefix.replace(".", "_")

    parser.add_argument(
        f"--{prefix}.must-contain-nonsilence",
        type=str2bool,
        dest=f"{p}_must_contain_nonsilence",
        default=rule.must_contain_nonsilence,
        help="""If true, for this endpointing rule to apply there must be
        nonsilence in the best-path traceback. For RNN-T decoding, a non-blank
        token is considered as non-silence""",
    )

    parser.add_argument(
        f"--{prefix}.min-trailing-silence",
        type=float,
        dest=f"{p}_min_trailing_silence",
        default=rule.min_trailing_silence,
        help="""This endpointing rule requires duration of trailing silence
        (in seconds) to be >= this value.""",
    )

    parser.add_argument(
        f"--{prefix}.min-utterance-length",
        type=float,
        dest=f"{p}_min_utterance_length",
        default=rule.min_utterance_length,
        help="""This endpointing rule requires utterance-length (in seconds)
        to be >= this value.""",
    )


def add_online_endpoint_arguments():
    """Add command line arguments to configure online endpointing.

    It provides the following commandline arguments:

        --endpoint.rule1.must-contain-nonsilence
        --endpoint.rule1.min_trailing_silence
        --endpoint.rule1.min_utterance_length

        --endpoint.rule2.must-contain-nonsilence
        --endpoint.rule2.min_trailing_silence
        --endpoint.rule2.min_utterance_length

        --endpoint.rule3.must-contain-nonsilence
        --endpoint.rule3.min_trailing_silence
        --endpoint.rule3.min_utterance_length

    You can add more rules if there is a need.
    """
    parser = argparse.ArgumentParser(
        description="Parameters for online endpoint detection",
        add_help=False,
    )

    config = OnlineEndpointConfig()
    _add_rule_arguments(parser, prefix="endpoint.rule1", rule=config.rule1)
    _add_rule_arguments(parser, prefix="endpoint.rule2", rule=config.rule2)
    _add_rule_arguments(parser, prefix="endpoint.rule3", rule=config.rule3)

    return parser


def _rule_activated(
    rule: OnlineEndpointRule,
    trailing_silence: float,
    utterance_length: float,
):
    """
    Args:
      rule:
        The rule to be checked.
      trailing_silence:
        Trailing silence in seconds.
      utterance_length:
        Number of frames in seconds decoded so far.
    Returns:
      Return True if the given rule is activated; return False otherwise.
    """
    contains_nonsilence = utterance_length > trailing_silence

    return (
        (contains_nonsilence or not rule.must_contain_nonsilence)
        and (trailing_silence > rule.min_trailing_silence)
        and (utterance_length > rule.min_utterance_length)
    )


def endpoint_detected(
    config: OnlineEndpointConfig,
    num_frames_decoded: int,
    trailing_silence_frames: int,
    frame_shift_in_seconds: float,
) -> bool:
    """
    Args:
      config:
        The endpoint config to be checked.
      num_frames_decoded:
        Number of frames decoded so far.
      trailing_silence_frames:
        Number of trailing silence frames.
      frame_shift_in_seconds:
        Frame shift in seconds.
    Returns:
      Return True if any rule in `config` is activated; return False otherwise.
    """
    utterance_length = num_frames_decoded * frame_shift_in_seconds
    trailing_silence = trailing_silence_frames * frame_shift_in_seconds

    if _rule_activated(config.rule1, trailing_silence, utterance_length):
        return True

    if _rule_activated(config.rule2, trailing_silence, utterance_length):
        return True

    if _rule_activated(config.rule3, trailing_silence, utterance_length):
        return True

    return False
