from typing import List

import k2
import torch
from stream import Stream, stack_states, unstack_states

from sherpa import (
    VALID_FAST_BEAM_SEARCH_METHOD,
    Lexicon,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_one_best,
    streaming_greedy_search,
)


class FastBeamSearch:
    def __init__(
        self,
        beam_search_params: dict,
        device: torch.device,
    ):
        """
        Args:
          beam_search_params
            Dictionary containing all the parameters for beam search.
          device:
            Device on which the computation will occur
        """

        decoding_method = beam_search_params["decoding_method"]
        assert (
            decoding_method in VALID_FAST_BEAM_SEARCH_METHOD
        ), f"{decoding_method} is not a valid search method"

        self.decoding_method = decoding_method
        self.rnnt_decoding_config = k2.RnntDecodingConfig(
            vocab_size=beam_search_params["vocab_size"],
            decoder_history_len=beam_search_params["context_size"],
            beam=beam_search_params["beam"],
            max_states=beam_search_params["max_states"],
            max_contexts=beam_search_params["max_contexts"],
        )
        if decoding_method == "fast_beam_search_nbest_LG":
            lexicon = Lexicon(beam_search_params["lang_dir"])
            self.word_table = lexicon.word_table
            lg_filename = beam_search_params["lang_dir"] / "LG.pt"
            self.decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            self.decoding_graph.scores *= beam_search_params["ngram_lm_scale"]
        else:
            self.decoding_graph = k2.trivial_graph(
                beam_search_params["vocab_size"] - 1, device
            )
        self.device = device
        self.context_size = beam_search_params["context_size"]
        self.beam_search_params = beam_search_params

    def init_stream(self, stream: Stream):
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
        """Run the model on the given stream list and do search with fast_beam_search
           method.
        Args:
          server:
            An instance of `StreamingServer`.
          stream_list:
            A list of streams to be processed. It is changed in-place.
            That is, the attribute `states` and `hyp` are
            updated in-place.
        """
        model = server.model
        device = model.device
        # Note: chunk_length is in frames before subsampling
        chunk_length = server.chunk_length
        batch_size = len(stream_list)
        chunk_length_pad = server.chunk_length_pad
        state_list, feature_list = [], []
        processed_frames_list, rnnt_decoding_streams_list = [], []

        rnnt_decoding_config = self.rnnt_decoding_config
        for s in stream_list:
            rnnt_decoding_streams_list.append(s.rnnt_decoding_stream)
            state_list.append(s.states)
            processed_frames_list.append(s.processed_frames)
            f = s.features[:chunk_length_pad]
            s.features = s.features[chunk_length:]
            s.processed_frames += chunk_length

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

        num_processed_frames = torch.tensor(
            processed_frames_list,
            device=device,
        )

        (
            encoder_out,
            encoder_out_lens,
            next_states,
        ) = model.encoder_streaming_forward(
            features=features,
            features_length=features_length,
            num_processed_frames=num_processed_frames,
            states=states,
        )

        processed_lens = (num_processed_frames >> 2) + encoder_out_lens
        if self.decoding_method == "fast_beam_search_nbest":
            next_hyp_list, next_trailing_blank_frames = fast_beam_search_nbest(
                model=model,
                encoder_out=encoder_out,
                processed_lens=processed_lens,
                rnnt_decoding_config=rnnt_decoding_config,
                rnnt_decoding_streams_list=rnnt_decoding_streams_list,
                num_paths=self.beam_search_params["num_paths"],
                nbest_scale=self.beam_search_params["nbest_scale"],
                use_double_scores=True,
                temperature=self.beam_search_params["temperature"],
            )
        elif self.decoding_method == "fast_beam_search_nbest_LG":
            (
                next_hyp_list,
                next_trailing_blank_frames,
            ) = fast_beam_search_nbest_LG(
                model=model,
                encoder_out=encoder_out,
                processed_lens=processed_lens,
                rnnt_decoding_config=rnnt_decoding_config,
                rnnt_decoding_streams_list=rnnt_decoding_streams_list,
                num_paths=self.beam_search_params["num_paths"],
                nbest_scale=self.beam_search_params["nbest_scale"],
                use_double_scores=True,
                temperature=self.beam_search_params["temperature"],
            )
        elif self.decoding_method == "fast_beam_search":
            (
                next_hyp_list,
                next_trailing_blank_frames,
            ) = fast_beam_search_one_best(
                model=model,
                encoder_out=encoder_out,
                processed_lens=processed_lens,
                rnnt_decoding_config=rnnt_decoding_config,
                rnnt_decoding_streams_list=rnnt_decoding_streams_list,
            )
        else:
            raise NotImplementedError(
                f"{self.decoding_method} is not implemented"
            )

        next_state_list = unstack_states(next_states)
        for i, s in enumerate(stream_list):
            s.states = next_state_list[i]
            s.hyp = next_hyp_list[i]
            s.num_trailing_blank_frames = next_trailing_blank_frames[i]

    def get_texts(self, stream: Stream) -> str:
        """
        Return text after decoding
        Args:
          stream:
            Stream to be processed.
        """
        if self.decoding_method == "fast_beam_search_nbest_LG":
            result = [self.word_table[i] for i in stream.hyp]
            result = " ".join(result)
        else:
            result = self.sp.decode(stream.hyp)

        return result


class GreedySearch:
    def __init__(
        self,
        model: "RnntConvEmformerModel",
        beam_search_params: dict,
        device: torch.device,
    ):
        """
        Args:
          model:
            RNN-T model decoder model
          beam_search_params:
            Dictionary containing all the parameters for beam search.
          device:
            Device on which the computation will occur
        """
        self.device = device
        self.beam_search_params = beam_search_params
        self.device = device

        decoder_input = torch.tensor(
            [
                [self.beam_search_params["blank_id"]]
                * self.beam_search_params["context_size"]
            ],
            device=self.device,
            dtype=torch.int64,
        )

        initial_decoder_out = model.decoder_forward(decoder_input)
        self.initial_decoder_out = model.forward_decoder_proj(
            initial_decoder_out.squeeze(1)
        )

    def init_stream(self, stream: Stream):
        """
        Attributes to add to each stream
        """
        stream.decoder_out = self.initial_decoder_out
        stream.hyp = [
            self.beam_search_params["blank_id"]
        ] * self.beam_search_params["context_size"]

    @torch.no_grad()
    def process(
        self,
        server: "StreamingServer",
        stream_list: List[Stream],
    ) -> None:
        """Run the model on the given stream list and do search with greedy_search
           method.
        Args:
          server:
            An instance of `StreamingServer`.
          stream_list:
            A list of streams to be processed. It is changed in-place.
            That is, the attribute `states` and `hyp` are
            updated in-place.
        """
        model = server.model
        device = model.device
        # Note: chunk_length is in frames before subsampling
        chunk_length = server.chunk_length
        batch_size = len(stream_list)
        chunk_length_pad = server.chunk_length_pad
        state_list, feature_list = [], []
        decoder_out_list, hyp_list = [], []
        processed_frames_list = []
        num_trailing_blank_frames_list = []

        for s in stream_list:
            decoder_out_list.append(s.decoder_out)
            hyp_list.append(s.hyp)
            state_list.append(s.states)
            processed_frames_list.append(s.processed_frames)
            f = s.features[:chunk_length_pad]
            s.features = s.features[chunk_length:]
            s.processed_frames += chunk_length

            b = torch.cat(f, dim=0)
            feature_list.append(b)

            num_trailing_blank_frames_list.append(s.num_trailing_blank_frames)

        features = torch.stack(feature_list, dim=0).to(device)
        states = stack_states(state_list)
        decoder_out = torch.cat(decoder_out_list, dim=0)

        features_length = torch.full(
            (batch_size,),
            fill_value=features.size(1),
            device=device,
            dtype=torch.int64,
        )

        num_processed_frames = torch.tensor(
            processed_frames_list,
            device=device,
        )

        (
            encoder_out,
            encoder_out_lens,
            next_states,
        ) = model.encoder_streaming_forward(
            features=features,
            features_length=features_length,
            num_processed_frames=num_processed_frames,
            states=states,
        )

        # Note: It does not return the next_encoder_out_len since
        # there are no paddings for streaming ASR. Each stream
        # has the same input number of frames, i.e., server.chunk_length.
        (
            next_decoder_out,
            next_hyp_list,
            next_trailing_blank_frames,
        ) = streaming_greedy_search(
            model=model,
            encoder_out=encoder_out,
            decoder_out=decoder_out,
            hyps=hyp_list,
            num_trailing_blank_frames=num_trailing_blank_frames_list,
        )

        next_decoder_out_list = next_decoder_out.split(1)

        next_state_list = unstack_states(next_states)
        for i, s in enumerate(stream_list):
            s.states = next_state_list[i]
            s.decoder_out = next_decoder_out_list[i]
            s.hyp = next_hyp_list[i]
            s.num_trailing_blank_frames = next_trailing_blank_frames[i]

    def get_texts(self, stream: Stream) -> str:
        """
        Return text after decoding
        Args:
          stream:
            Stream to be processed.
        """
        return self.sp.decode(
            stream.hyp[self.beam_search_params["context_size"] :]
        )
