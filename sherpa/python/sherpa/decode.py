# Copyright      2022  Xiaomi Corp.        (authors: Wei Kang)
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

from typing import List, Union

import k2
import torch
from _sherpa import RnntModel
from icefall.decode import Nbest

VALID_FAST_BEAM_SEARCH_METHOD = ["fast_beam_search_nbest", "fast_beam_search"]


def fast_beam_search_nbest(
    model: RnntModel,
    encoder_out: torch.Tensor,
    processed_lens: torch.Tensor,
    rnnt_decoding_config: k2.RnntDecodingConfig,
    rnnt_decoding_streams_list: List[k2.RnntDecodingStream],
    num_paths: int,
    nbest_scale: float = 0.5,
    use_double_scores: bool = True,
    temperature: float = 1.0,
) -> List[List[int]]:
    """It limits the maximum number of symbols per frame to 1.

    The process to get the results is:
     - (1) Use fast beam search to get a lattice
     - (2) Select `num_paths` paths from the lattice using k2.random_paths()
     - (3) Unique the selected paths
     - (4) Intersect the selected paths with the lattice and compute the
           shortest path from the intersection result
     - (5) The path with the largest score is used as the decoding output.

    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      processed_lens:
        A 1-D tensor containing the valid frames before padding that have been
        processed by encoder network until now. For offline recognition, it equals
        to ``encoder_out_lens`` of encoder outputs. For online recognition, it is
        the cumulative sum of ``encoder_out_lens`` of previous chunks (including
        current chunk). Its dtype is `torch.kLong` and its shape is `(batch_size,)`.
      rnnt_decoding_config:
        The configuration of Fsa based RNN-T decoding, refer to
        https://k2-fsa.github.io/k2/python_api/api.html#rnntdecodingconfig for more
        details.
      rnnt_decoding_streams_list:
        A list containing the RnntDecodingStream for each sequences, its size is
        ``encoder_out.size(0)``. It stores the decoding graph, internal decoding
        states and partial results.
      num_paths:
        Number of paths to extract from the decoded lattice.
      nbest_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
      use_double_scores:
        True to use double precision for computation. False to use
        single precision.
      temperature:
        Softmax temperature.
    Returns:
      Return the decoded result.
    """

    lattice = fast_beam_search(
        model=model,
        encoder_out=encoder_out,
        processed_lens=processed_lens,
        rnnt_decoding_config=rnnt_decoding_config,
        rnnt_decoding_streams_list=rnnt_decoding_streams_list,
        temperature=temperature,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    # at this point, nbest.fsa.scores are all zeros.
    nbest = nbest.intersect(lattice)
    # Now nbest.fsa.scores contains acoustic scores

    max_indexes = nbest.tot_scores().argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)

    hyps = get_texts(best_path)

    return hyps


def fast_beam_search_one_best(
    model: RnntModel,
    encoder_out: torch.Tensor,
    processed_lens: torch.Tensor,
    rnnt_decoding_config: k2.RnntDecodingConfig,
    rnnt_decoding_streams_list: List[k2.RnntDecodingStream],
    temperature: float = 1.0,
) -> List[List[int]]:
    """It limits the maximum number of symbols per frame to 1.

    A lattice is first obtained using fast beam search, and then
    the shortest path within the lattice is used as the final output.

    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      processed_lens:
        A 1-D tensor containing the valid frames before padding that have been
        processed by encoder network until now. For offline recognition, it equals
        to ``encoder_out_lens`` of encoder outputs. For online recognition, it is
        the cumulative sum of ``encoder_out_lens`` of previous chunks (including
        current chunk). Its dtype is `torch.kLong` and its shape is `(batch_size,)`.
      rnnt_decoding_config:
        The configuration of Fsa based RNN-T decoding, refer to
        https://k2-fsa.github.io/k2/python_api/api.html#rnntdecodingconfig for more
        details.
      rnnt_decoding_streams_list:
        A list containing the RnntDecodingStream for each sequences, its size is
        ``encoder_out.size(0)``. It stores the decoding graph, internal decoding
        states and partial results.
      temperature:
        Softmax temperature.
    Returns:
      Return the decoded result.
    """
    lattice = fast_beam_search(
        model=model,
        encoder_out=encoder_out,
        processed_lens=processed_lens,
        rnnt_decoding_config=rnnt_decoding_config,
        rnnt_decoding_streams_list=rnnt_decoding_streams_list,
        temperature=temperature,
    )

    best_path = one_best_decoding(lattice)
    hyps = get_texts(best_path)
    return hyps


def fast_beam_search(
    model: RnntModel,
    encoder_out: torch.Tensor,
    processed_lens: torch.Tensor,
    rnnt_decoding_config: k2.RnntDecodingConfig,
    rnnt_decoding_streams_list: List[k2.RnntDecodingStream],
    temperature: float = 1.0,
) -> k2.Fsa:
    """It limits the maximum number of symbols per frame to 1.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a LG.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      processed_lens:
        A 1-D tensor containing the valid frames before padding that have been
        processed by encoder network until now. For offline recognition, it equals
        to ``encoder_out_lens`` of encoder outputs. For online recognition, it is
        the cumulative sum of ``encoder_out_lens`` of previous chunks (including
        current chunk). Its dtype is `torch.kLong` and its shape is `(batch_size,)`.
      rnnt_decoding_config:
        The configuration of Fsa based RNN-T decoding, refer to
        https://k2-fsa.github.io/k2/python_api/api.html#rnntdecodingconfig for more
        details.
      rnnt_decoding_streams_list:
        A list containing the RnntDecodingStream for each sequences, its size is
        ``encoder_out.size(0)``. It stores the decoding graph, internal decoding
        states and partial results.
      temperature:
        Softmax temperature.
    Returns:
      Return an FsaVec with axes [utt][state][arc] containing the decoded
      lattice. Note: When the input graph is a TrivialGraph, the returned
      lattice is actually an acceptor.
    """
    assert encoder_out.ndim == 3

    B, T, C = encoder_out.shape

    decoding_streams = k2.RnntDecodingStreams(
        rnnt_decoding_streams_list, rnnt_decoding_config
    )

    encoder_out = model.forward_encoder_proj(encoder_out)

    for t in range(T):
        # shape is a RaggedShape of shape (B, context)
        # contexts is a Tensor of shape (shape.NumElements(), context_size)
        shape, contexts = decoding_streams.get_contexts()
        # `nn.Embedding()` in torch below v1.7.1 supports only torch.int64
        contexts = contexts.to(torch.int64)
        # decoder_out is of shape (shape.NumElements(), 1, decoder_out_dim)
        decoder_out = model.decoder_forward(contexts)
        decoder_out = model.forward_decoder_proj(decoder_out).squeeze(1)
        # current_encoder_out is of shape
        # (shape.NumElements(), joiner_dim)
        # fmt: off
        current_encoder_out = torch.index_select(
            encoder_out[:, t], 0, shape.row_ids(1).to(torch.int64)
        )
        # fmt: on
        logits = model.joiner_forward(
            current_encoder_out,
            decoder_out,
        )
        log_probs = (logits / temperature).log_softmax(dim=-1)
        decoding_streams.advance(log_probs)
    decoding_streams.terminate_and_flush_to_streams()
    lattice = decoding_streams.format_output(processed_lens.tolist())

    return lattice


def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
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
    if return_ragged:
        return aux_labels
    else:
        return aux_labels.tolist()


def one_best_decoding(
    lattice: k2.Fsa,
    use_double_scores: bool = True,
) -> k2.Fsa:
    """Get the best path from a lattice.

    Args:
      lattice:
        The decoding lattice returned by :func:`get_lattice`.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
    Return:
      An FsaVec containing linear paths.
    """
    best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
    return best_path
