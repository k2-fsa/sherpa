import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

Pathlike = Union[str, Path]


def setup_logger(
    log_filename: Pathlike,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    log_filename = f"{log_filename}-{date_time}.txt"

    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


@dataclass
class FastBeamSearchResults:
    # hyps[i] is the recognition results for the i-th utterance.
    # It may contain either token IDs or word IDs depending on the actual
    # decoding method.
    hyps: List[List[int]]

    # Number of trailing blank for each utterance in the batch
    num_trailing_blanks: List[int]

    # Decoded token IDs for each utterance in the batch
    tokens: List[List[int]]

    # timestamps[i][k] contains the frame number on which tokens[i][k]
    # is decoded
    timestamps: List[List[int]]


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def encode_contexts(
    modeling_unit: str,
    contexts: List[str],
    sp: Optional["SentencePieceProcessor"] = None,  # noqa
    tokens_table: Optional[Dict[str, int]] = None,
) -> List[List[int]]:
    """
    Encode the given contexts (a list of string) to a list of a list of token
    ids.

    Args:
      modeling_unit:
        The valid values are bpe, char, bpe+char.
        Note: char here means characters in CJK languages, not English like
        languages.
      contexts:
        The given contexts list (a list of string).
      sp:
        An instance of SentencePieceProcessor.
      tokens_table:
        The tokens_table containing the tokens and the corresponding ids.
    Returns:
      Return the contexts_list, it is a list of a list of token ids.
    """
    contexts_list = []
    if "bpe" in modeling_unit:
        assert sp is not None
    if "char" in modeling_unit:
        assert tokens_table is not None
        assert len(tokens_table) > 0, len(tokens_table)

    if "char" == modeling_unit:
        for context in contexts:
            assert " " not in context
            ids = [
                tokens_table[txt]
                if txt in tokens_table
                else tokens_table["<unk>"]
                for txt in context
            ]
            contexts_list.append(ids)
    elif "bpe" == modeling_unit:
        contexts_list = sp.encode(contexts, out_type=int)
    else:
        assert modeling_unit == "bpe+char", modeling_unit

        # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        pattern = re.compile(r"([\u4e00-\u9fff])")
        for context in contexts:
            # Example:
            #   txt   = "你好 ITS'S OKAY 的"
            #   chars = ["你", "好", " ITS'S OKAY ", "的"]
            chars = pattern.split(context.upper())
            mix_chars = [w for w in chars if len(w.strip()) > 0]
            ids = []
            for ch_or_w in mix_chars:
                # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
                if pattern.fullmatch(ch_or_w) is not None:
                    ids.append(
                        tokens_table[ch_or_w]
                        if ch_or_w in tokens_table
                        else tokens_table["<unk>"]
                    )
                # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
                # encode ch_or_w using bpe_model.
                else:
                    for p in sp.encode_as_pieces(ch_or_w):
                        ids.append(
                            tokens_table[p]
                            if p in tokens_table
                            else tokens_table["<unk>"]
                        )
        contexts_list.append(ids)
    return contexts_list
