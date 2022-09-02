import argparse
from pathlib import Path
from typing import List, Tuple, Union

import k2


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


def count_num_trailing_zeros(labels: List[int]):
    """Return the number of trailing zeros in labels."""
    n = 0
    for v in reversed(labels):
        if v != 0:
            break
        else:
            n += 1
    return n


def get_texts_and_num_trailing_blanks(
    best_paths: k2.Fsa,
) -> Tuple[Union[List[List[int]], k2.RaggedTensor], List[int]]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
    Returns:
      Return a tuple containing
        - a list of lists of int, containing the label sequences we decoded.
        - number of trailing blank frames
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2

    # Now count number of trailing frames
    labels_shape = best_paths.arcs.shape().remove_axis(1)
    labels_list = k2.RaggedTensor(
        labels_shape, best_paths.labels.contiguous()
    ).tolist()

    num_trailing_blanks = []
    for labels in labels_list:
        # [:-1] to remove the last -1
        num_trailing_blanks.append(count_num_trailing_zeros(labels[:-1]))

    return aux_labels.tolist(), num_trailing_blanks


def add_beam_search_arguments():
    parser = argparse.ArgumentParser(
        description="Parameters for beam search", add_help=False
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Decoding method to use. Possible values are:
        greedy_search, fast_beam_search, fast_beam_search_nbest
          """,
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=10.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
           Used only when the decoding method is fast_beam_search_nbest,
           fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--num-active-paths",
        type=int,
        default=4,
        help="""Used only when decoding_method is modified_beam_search.
            It specifies number of active paths for each utterance. Due to
            merging paths with identical token sequences, the actual number
            may be less than "num_active_paths".""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="""Softmax temperature.
         The output of the model is (logits / temperature).log_softmax().
         """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is fast_beam_search.""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=32,
        help="""Used only when --decoding-method is fast_beam_search.""",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    return parser
