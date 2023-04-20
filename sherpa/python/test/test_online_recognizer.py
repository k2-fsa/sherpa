#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_online_recognizer_py

import unittest
import wave
from pathlib import Path

import torch
import torchaudio

import sherpa


def decode(
    recognizer: sherpa.OnlineRecognizer,
    s: sherpa.OnlineStream,
    samples: torch.Tensor,
):
    expected_sample_rate = 16000

    tail_padding = torch.zeros(int(16000 * 0.3), dtype=torch.float32)

    chunk = int(0.2 * expected_sample_rate)  # 0.2 seconds

    start = 0
    last_result = ""
    while start < samples.numel():
        end = start + chunk
        s.accept_waveform(expected_sample_rate, samples[start:end])
        start = end

        while recognizer.is_ready(s):
            recognizer.decode_stream(s)
            result = recognizer.get_result(s).text
            if last_result != result:
                last_result = result
                print(result)

    s.accept_waveform(expected_sample_rate, tail_padding)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)
        result = recognizer.get_result(s).text
        if last_result != result:
            last_result = result
            print(result)


d = "/tmp/icefall-models"
# Please refer to
# https://k2-fsa.github.io/sherpa/cpp/pretrained_models/online_transducer.html
# to download pre-trained models for testing
class TestOnlineRecognizer(unittest.TestCase):
    def test_icefall_asr_librispeech_conv_emformer_transducer_stateless2_2022_07_05(
        self,
    ):
        nn_model = f"{d}/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/exp/cpu-jit-epoch-30-avg-10-torch-1.10.0.pt"
        tokens = f"{d}/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/data/lang_bpe_500/tokens.txt"
        lg = f"{d}/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/data/lang_bpe_500/LG.pt"
        wave = f"{d}/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05/test_wavs/1089-134686-0001.wav"

        if not Path(nn_model).is_file():
            print(f"{nn_model} does not exist")
            print(
                "skipping test_icefall_asr_librispeech_conv_emformer_transducer_stateless2_2022_07_05()"
            )
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------greedy search--------------------")

        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="greedy_search",
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)
        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print("--------------------fast beam search--------------------")
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print(
            "--------------------fast beam search with LG--------------------"
        )
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
            lg=lg,
            ngram_lm_scale=0.01,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

    def test_icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming(
        self,
    ):
        nn_model = f"{d}/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/exp/cpu_jit_epoch_7_avg_1_torch.1.7.1.pt"
        tokens = f"{d}/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/data/lang_char/tokens.txt"
        lg = f"{d}/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/data/lang_char/LG.pt"
        wave = f"{d}/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/test_wavs/DEV_T0000000000.wav"

        if not Path(nn_model).is_file():
            print(f"{nn_model} does not exist")
            print(
                "skipping test_icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming()"
            )
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------greedy search--------------------")

        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="greedy_search",
            left_context=64,
            right_context=0,
            chunk_size=12,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)
        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
            left_context=64,
            right_context=0,
            chunk_size=12,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print("--------------------fast beam search--------------------")
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
            left_context=64,
            right_context=0,
            chunk_size=12,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print(
            "--------------------fast beam search with LG--------------------"
        )
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
            lg=lg,
            ngram_lm_scale=0.01,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
            left_context=64,
            right_context=0,
            chunk_size=12,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

    def test_icefall_asr_conv_emformer_transducer_stateless2_zh(
        self,
    ):
        nn_model = f"{d}/icefall-asr-conv-emformer-transducer-stateless2-zh/exp/cpu_jit-epoch-11-avg-1.pt"
        tokens = f"{d}/icefall-asr-conv-emformer-transducer-stateless2-zh/data/lang_char_bpe/tokens.txt"
        wave = f"{d}/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/test_wavs/DEV_T0000000000.wav"

        if not Path(nn_model).is_file():
            print(f"{nn_model} does not exist")
            print(
                "skipping test_icefall_asr_librispeech_conv_emformer_transducer_stateless2_2022_07_05()"
            )
            return

        if not Path(wave).is_file():
            print(f"{wave} does not exist")
            print(
                "skipping test_icefall_asr_librispeech_conv_emformer_transducer_stateless2_2022_07_05()"
            )
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------greedy search--------------------")

        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="greedy_search",
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)
        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print("--------------------fast beam search--------------------")
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

    def test_icefall_librispeech_streaming_pruned_transducer_stateless4_20220625(
        self,
    ):
        nn_model = f"{d}/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625/exp/cpu_jit-epoch-25-avg-3.pt"
        tokens = f"{d}/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625/data/lang_bpe_500/tokens.txt"
        lg = f"{d}/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625/data/lang_bpe_500/LG.pt"
        wave = f"{d}/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625/test_waves/1089-134686-0001.wav"

        if not Path(nn_model).is_file():
            print(f"{nn_model} does not exist")
            print(
                "skipping test_icefall_librispeech_streaming_pruned_transducer_stateless4_20220625()"
            )
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------greedy search--------------------")

        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="greedy_search",
            left_context=64,
            right_context=0,
            chunk_size=12,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)
        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
            left_context=64,
            right_context=0,
            chunk_size=12,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print("--------------------fast beam search--------------------")
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
            left_context=64,
            right_context=0,
            chunk_size=12,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print(
            "--------------------fast beam search with LG--------------------"
        )
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
            lg=lg,
            ngram_lm_scale=0.01,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
            left_context=64,
            right_context=0,
            chunk_size=12,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

    def test_cefall_asr_librispeech_lstm_transducer_stateless2_2022_09_03(
        self,
    ):
        encoder_model = f"{d}/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/encoder_jit_trace-iter-468000-avg-16.pt"
        decoder_model = f"{d}/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/decoder_jit_trace-iter-468000-avg-16.pt"
        joiner_model = f"{d}/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/exp/joiner_jit_trace-iter-468000-avg-16.pt"

        tokens = f"{d}/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/data/lang_bpe_500/tokens.txt"
        lg = f"{d}/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/data/lang_bpe_500/LG.pt"
        wave = f"{d}/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/test_wavs/1089-134686-0001.wav"

        if not Path(encoder_model).is_file():
            print(f"{encoder_model} does not exist")
            print(
                "skipping test_icefall_librispeech_streaming_pruned_transducer_stateless4_20220625()"
            )
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------greedy search--------------------")

        config = sherpa.OnlineRecognizerConfig(
            nn_model="",
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            joiner_model=joiner_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="greedy_search",
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)
        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model="",
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            joiner_model=joiner_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print("--------------------fast beam search--------------------")
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model="",
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            joiner_model=joiner_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print(
            "--------------------fast beam search with LG--------------------"
        )
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
            lg=lg,
            ngram_lm_scale=0.01,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model="",
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            joiner_model=joiner_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

    def test_icefall_asr_librispeech_pruned_stateless_emformer_rnnt2_2022_06_01(
        self,
    ):
        nn_model = f"{d}/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01/exp/cpu_jit-epoch-39-avg-6-use-averaged-model-1.pt"
        tokens = f"{d}/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01/data/lang_bpe_500/tokens.txt"
        lg = f"{d}/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01/data/lang_bpe_500/LG.pt"
        wave = f"{d}/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01/test_wavs/1089-134686-0001.wav"

        if not Path(nn_model).is_file():
            print(f"{nn_model} does not exist")
            print(
                "skipping test_icefall_asr_librispeech_conv_emformer_transducer_stateless2_2022_07_05()"
            )
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------greedy search--------------------")

        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="greedy_search",
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)
        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print("--------------------fast beam search--------------------")
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print(
            "--------------------fast beam search with LG--------------------"
        )
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
            lg=lg,
            ngram_lm_scale=0.01,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

    def test_k2fsa_zipformer_chinese_english_mixed(self):
        nnr_model = f"{d}/k2fsa-zipformer-chinese-english-mixed/exp/cpu_jit.pt"
        tokens = f"{d}/k2fsa-zipformer-chinese-english-mixed/data/lang_char_bpe/tokens.txt"
        wave = f"{d}/k2fsa-zipformer-chinese-english-mixed/test_wavs/0.wav"

        if not Path(encoder_model).is_file():
            print(f"{nn_model} does not exist")
            print("skipping test_k2fsa_zipformer_chinese_english_mixed()")
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------greedy search--------------------")

        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="greedy_search",
            chunk_size=32,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)
        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
            chunk_size=32,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print("--------------------fast beam search--------------------")
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
            chunk_size=32,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

    def test_icefall_asr_librispeech_pruned_transducer_stateless7_streaming_2022_12_29(
        self,
    ):
        nn_model = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/cpu_jit.pt"
        tokens = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/tokens.txt"
        lg = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/LG.pt"
        wave = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1089-134686-0001.wav"

        if not Path(encoder_model).is_file():
            print(f"{nn_model} does not exist")
            print(
                "skipping test_icefall_asr_librispeech_pruned_transducer_stateless7_streaming_2022_12_29()"
            )
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------greedy search--------------------")

        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="greedy_search",
            chunk_size=32,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)
        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
            chunk_size=32,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print("--------------------fast beam search--------------------")
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
            chunk_size=32,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)

        print(
            "--------------------fast beam search with LG--------------------"
        )
        fast_beam_search_config = sherpa.FastBeamSearchConfig(
            beam=20.0,
            max_states=64,
            max_contexts=8,
            allow_partial=True,
            lg=lg,
            ngram_lm_scale=0.01,
        )
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="fast_beam_search",
            fast_beam_search_config=fast_beam_search_config,
            chunk_size=32,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    unittest.main()
