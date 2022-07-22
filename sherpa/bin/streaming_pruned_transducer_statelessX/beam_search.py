from typing import List

import k2
import torch
from decode import Stream
from streaming_server import StreamingServer

from sherpa import fast_beam_search_one_best, streaming_greedy_search


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

    def get_attribute_stream(self, stream: "Stream"):
        """
        Attributes to add to each stream
        """
        stream.__dict__["rnnt_decoding_stream"] = k2.RnntDecodingStream(
            self.decoding_graph
        )
        stream.__dict__["hyp"] = []

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
        # Note: chunk_length is in frames before subsampling
        chunk_length = server.chunk_length
        subsampling_factor = server.subsampling_factor
        # Note: chunk_size, left_context and right_context are in frames
        # after subsampling
        chunk_size = server.decode_chunk_size
        left_context = server.decode_left_context
        right_context = server.decode_right_context

        batch_size = len(stream_list)

        state_list = []
        feature_list = []
        processed_frames_list = []

        rnnt_decoding_streams_list = []
        rnnt_decoding_config = self.rnnt_decoding_config
        for s in stream_list:
            rnnt_decoding_streams_list.append(s.rnnt_decoding_stream)
            state_list.append(s.states)
            processed_frames_list.append(s.processed_frames)
            f = s.features[:chunk_length]
            s.features = s.features[chunk_size * subsampling_factor :]
            b = torch.cat(f, dim=0)
            feature_list.append(b)

        features = torch.stack(feature_list, dim=0).to(self.device)

        states = [
            torch.stack([x[0] for x in state_list], dim=2),
            torch.stack([x[1] for x in state_list], dim=2),
        ]

        features_length = torch.full(
            (batch_size,),
            fill_value=features.size(1),
            device=self.device,
            dtype=torch.int64,
        )

        processed_frames = torch.tensor(
            processed_frames_list, device=self.device
        )

        (
            encoder_out,
            encoder_out_lens,
            next_states,
        ) = model.encoder_streaming_forward(
            features=features,
            features_length=features_length,
            states=states,
            processed_frames=processed_frames,
            left_context=left_context,
            right_context=right_context,
        )

        processed_lens = processed_frames + encoder_out_lens
        next_hyp_list = fast_beam_search_one_best(
            model=model,
            encoder_out=encoder_out,
            processed_lens=processed_lens,
            rnnt_decoding_config=rnnt_decoding_config,
            rnnt_decoding_streams_list=rnnt_decoding_streams_list,
        )

        next_state_list = [
            torch.unbind(next_states[0], dim=2),
            torch.unbind(next_states[1], dim=2),
        ]

        for i, s in enumerate(stream_list):
            s.states = [next_state_list[0][i], next_state_list[1][i]]
            s.processed_frames += encoder_out_lens[i]
            s.hyp = next_hyp_list[i]

    def get_texts(self, stream):
        """
        Return text after decoding
        Args:
          stream:
            Stream to be processed.
        """
        if hasattr(self, "sp"):
            result = self.sp.decode(stream.hyp)
        else:
            result = [self.token_table[i] for i in stream.hyp]
        return result


class GreedySearch:
    def __init__(self, model, device):
        """
        Args:
          model:
            RNN-T model decoder model
          device:
            Device on which the computation will occur
        """

        self.blank_id = self.model.blank_id
        self.context_size = self.model.context_size
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

    def get_attribute_stream(self, stream: "Stream"):
        """
        Attributes to add to each stream
        """
        stream.__dict__["decoder_out"] = self.initial_decoder_out
        stream.__dict__["hyp"] = [self.blank_id] * self.context_size

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
        subsampling_factor = server.subsampling_factor
        # Note: chunk_size, left_context and right_context are in frames
        # after subsampling
        chunk_size = server.decode_chunk_size
        left_context = server.decode_left_context
        right_context = server.decode_right_context

        batch_size = len(stream_list)

        state_list, feature_list, processed_frames_list = [], [], []
        decoder_out_list, hyp_list = [], []

        for s in stream_list:
            decoder_out_list.append(s.decoder_out)
            hyp_list.append(s.hyp)
            state_list.append(s.states)
            processed_frames_list.append(s.processed_frames)
            f = s.features[:chunk_length]
            s.features = s.features[chunk_size * subsampling_factor :]
            b = torch.cat(f, dim=0)
            feature_list.append(b)

        features = torch.stack(feature_list, dim=0).to(device)

        states = [
            torch.stack([x[0] for x in state_list], dim=2),
            torch.stack([x[1] for x in state_list], dim=2),
        ]

        decoder_out = torch.cat(decoder_out_list, dim=0)

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
        ) = model.encoder_streaming_forward(
            features=features,
            features_length=features_length,
            states=states,
            processed_frames=processed_frames,
            left_context=left_context,
            right_context=right_context,
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

        next_state_list = [
            torch.unbind(next_states[0], dim=2),
            torch.unbind(next_states[1], dim=2),
        ]
        next_decoder_out_list = next_decoder_out.split(1)

        for i, s in enumerate(stream_list):
            s.states = [next_state_list[0][i], next_state_list[1][i]]
            s.processed_frames += encoder_out_lens[i]
            s.decoder_out = next_decoder_out_list[i]
            s.hyp = next_hyp_list[i]

    def get_texts(self, stream):
        """
        Return text after decoding
        Args:
          stream:
            Stream to be processed.
        """
        if hasattr(self, "sp"):
            result = self.sp.decode(stream.hyp[self.context_size :])  # noqa
        else:
            result = [
                self.token_table[i] for i in stream.hyp[self.context_size :]
            ]  # noqa
        return result
