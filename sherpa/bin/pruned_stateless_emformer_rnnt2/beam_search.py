from typing import List

import k2
import torch
from decode import Stream, stack_states, unstack_states

from sherpa import (
    Hypotheses,
    Hypothesis,
    fast_beam_search_one_best,
    streaming_greedy_search,
    streaming_modified_beam_search,
)


class FastBeamSearch:
    def __init__(
        self, vocab_size, context_size, beam, max_states, max_contexts, device
    ):
        """
        Args:
          vocab_size:
            Vocabularize of the BPE
          context_size:
            Context size of the RNN-T decoder model.
          beam:
            The beam for fast_beam_search decoding.
          max_states:
            The max_states for fast_beam_search decoding.
          max_contexts:
            The max_contexts for fast_beam_search decoding.
          device:
            Device on which the computation will occur
        """
        self.rnnt_decoding_config = k2.RnntDecodingConfig(
            vocab_size=vocab_size,
            decoder_history_len=context_size,
            beam=beam,
            max_states=max_states,
            max_contexts=max_contexts,
        )
        self.decoding_graph = k2.trivial_graph(vocab_size - 1, device)
        self.device = device
        self.context_size = context_size

    def init_stream(self, stream: "Stream"):
        """
        Attributes to add to each stream
        """
        stream.rnnt_decoding_stream = k2.RnntDecodingStream(self.decoding_graph)
        stream.hyp = []

    @torch.no_grad()
    def process(
        self,
        server: "StreamingServer",
        stream_list: List[Stream],
    ) -> None:
        """Run the model on the given stream list and do search with given decoding
           method.
        Args:
          server:
            An instance of `StreamingServer`.
          stream_list:
            A list of streams to be processed. It is changed in-place.
            That is, the attribute `states`, `decoder_out`, and `hyp` are
            updated in-place.
        """
        model = server.model
        device = model.device
        # Note: chunk_length is in frames before subsampling
        chunk_length = server.chunk_length
        segment_length = server.segment_length
        batch_size = len(stream_list)

        state_list, feature_list = [], []
        processed_frames_list, rnnt_decoding_streams_list = [], []

        rnnt_decoding_config = self.rnnt_decoding_config
        for s in stream_list:
            rnnt_decoding_streams_list.append(s.rnnt_decoding_stream)

            state_list.append(s.states)
            processed_frames_list.append(s.processed_frames)
            f = s.features[:chunk_length]
            s.features = s.features[segment_length:]
            b = torch.cat(f, dim=0)
            feature_list.append(b)

        features = torch.stack(feature_list, dim=0).to(device)
        states = stack_states(state_list)

        features_length = torch.full(
            (batch_size,),
            fill_value=features.size(1),
            device=device,
            dtype=torch.int64,
        )

        processed_frames = torch.tensor(processed_frames_list, device=device)

        (
            encoder_out,
            encoder_out_lens,
            next_states,
        ) = model.encoder_streaming_forward(  # noqa
            features=features,
            features_length=features_length,
            states=states,
        )

        processed_lens = processed_frames + encoder_out_lens
        next_hyp_list = fast_beam_search_one_best(
            model=model,
            encoder_out=encoder_out,
            processed_lens=processed_lens,
            rnnt_decoding_config=rnnt_decoding_config,
            rnnt_decoding_streams_list=rnnt_decoding_streams_list,
        )

        next_state_list = unstack_states(next_states)
        for i, s in enumerate(stream_list):
            s.states = next_state_list[i]
            s.processed_frames += encoder_out_lens[i]
            s.hyp = next_hyp_list[i]

    def get_texts(self, stream):
        """
        Return text after decoding
        Args:
          stream:
            Stream to be processed.
        """
        return self.sp.decode(stream.hyp)


class GreedySearch:
    def __init__(self, model, device):
        """
        Args:
          model:
            RNN-T model decoder model
          device:
            Device on which the computation will occur
        """

        self.blank_id = model.blank_id
        self.context_size = model.context_size
        self.device = device

        decoder_input = torch.tensor(
            [[self.blank_id] * self.context_size],
            device=self.device,
            dtype=torch.int64,
        )
        initial_decoder_out = model.decoder_forward(decoder_input)
        self.initial_decoder_out = model.forward_decoder_proj(
            initial_decoder_out.squeeze(1)
        )

    def init_stream(self, stream: "Stream"):
        """
        Attributes to add to each stream
        """
        stream.decoder_out = self.initial_decoder_out
        stream.hyp = [self.blank_id] * self.context_size

    @torch.no_grad()
    def process(
        self,
        server: "StreamingServer",
        stream_list: List[Stream],
    ) -> None:
        """Run the model on the given stream list and do search with given decoding
           method.
        Args:
          server:
            An instance of `StreamingServer`.
          stream_list:
            A list of streams to be processed. It is changed in-place.
            That is, the attribute `states`, `decoder_out`, and `hyp` are
            updated in-place.
        """
        model = server.model
        device = model.device
        # Note: chunk_length is in frames before subsampling
        chunk_length = server.chunk_length
        batch_size = len(stream_list)
        segment_length = server.segment_length

        state_list, feature_list = [], []
        decoder_out_list, hyp_list = [], []

        for s in stream_list:
            decoder_out_list.append(s.decoder_out)
            hyp_list.append(s.hyp)

            state_list.append(s.states)

            f = s.features[:chunk_length]
            s.features = s.features[segment_length:]
            b = torch.cat(f, dim=0)
            feature_list.append(b)

        features = torch.stack(feature_list, dim=0).to(device)
        states = stack_states(state_list)

        decoder_out = torch.cat(decoder_out_list, dim=0)

        features_length = torch.full(
            (batch_size,),
            fill_value=features.size(1),
            device=device,
            dtype=torch.int64,
        )

        (
            encoder_out,
            _,
            next_states,
        ) = model.encoder_streaming_forward(  # noqa
            features=features,
            features_length=features_length,
            states=states,
        )

        # Note: It does not return the next_encoder_out_len since
        # there are no paddings for streaming ASR. Each stream
        # has the same input number of frames, i.e., server.chunk_length.
        next_decoder_out, next_hyp_list = streaming_greedy_search(
            model=model,
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            hyps=hyp_list,
        )

        next_decoder_out_list = next_decoder_out.split(1)

        next_state_list = unstack_states(next_states)
        for i, s in enumerate(stream_list):
            s.states = next_state_list[i]
            s.decoder_out = next_decoder_out_list[i]
            s.hyp = next_hyp_list[i]

    def get_texts(self, stream):
        """
        Return text after decoding
        Args:
          stream:
            Stream to be processed.
        """
        hyp = stream.hyp[self.context_size :]  # noqa
        return self.sp.decode(hyp)


class ModifiedBeamSearch:
    def __init__(self, blank_id, context_size):
        self.blank_id = blank_id
        self.context_size = context_size

    def init_stream(self, stream: "Stream"):
        """
        Attributes to add to each stream
        """
        hyp = [self.blank_id] * self.context_size
        stream.hyps = Hypotheses([Hypothesis(ys=hyp, log_prob=0.0)])

    def process(
        self,
        server: "StreamingServer",
        stream_list: List[Stream],
    ) -> None:
        """Run the model on the given stream list and do modified_beam_search.
        Args:
          server:
            An instance of `StreamingServer`.
          stream_list:
            A list of streams to be processed. It is changed in-place.
            That is, the attribute `states` and `hyps` are
            updated in-place.
        """
        model = server.model
        device = model.device

        segment_length = server.segment_length
        chunk_length = server.chunk_length

        batch_size = len(stream_list)

        state_list = []
        hyps_list = []
        feature_list = []
        for s in stream_list:
            state_list.append(s.states)
            hyps_list.append(s.hyps)

            f = s.features[:chunk_length]
            s.features = s.features[segment_length:]

            b = torch.cat(f, dim=0)
            feature_list.append(b)

        features = torch.stack(feature_list, dim=0).to(device)
        states = stack_states(state_list)

        features_length = torch.full(
            (batch_size,),
            fill_value=features.size(1),
            device=device,
            dtype=torch.int64,
        )

        (encoder_out, _, next_states) = model.encoder_streaming_forward(
            features=features,
            features_length=features_length,
            states=states,
        )
        # Note: There are no paddings for streaming ASR. Each stream
        # has the same input number of frames, i.e., server.chunk_length.
        next_hyps_list = streaming_modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            hyps=hyps_list,
        )

        next_state_list = unstack_states(next_states)
        for i, s in enumerate(stream_list):
            s.states = next_state_list[i]
            s.hyps = next_hyps_list[i]

    def get_texts(self, stream):
        hyp = stream.hyps.get_most_probable(True).ys[self.context_size :]
        return self.sp.decode(hyp)
