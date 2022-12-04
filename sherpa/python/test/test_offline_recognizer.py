#!/usr/bin/env python3
# noqa
# To run this single test, use
#
#  ctest --verbose -R  test_offline_recognizer_py

import unittest
from pathlib import Path

import sherpa


d = "/tmp/icefall-models"
# Please refer to
# https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc.html
# and
# https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_transducer.html
# to download pre-trained models for testing
class TestOfflineRecognizer(unittest.TestCase):
    def test_icefall_ctc_model(self):
        nn_model = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt"
        tokens = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/tokens.txt"
        wave1 = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav"
        wave2 = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav"

        if not Path(nn_model).is_file():
            print("skipping test_icefall_ctc_model()")
            return

        print()
        print("test_icefall_ctc_model()")

        feat_config = sherpa.FeatureConfig()

        feat_config.fbank_opts.frame_opts.samp_freq = 16000
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        config = sherpa.OfflineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
        )

        recognizer = sherpa.OfflineRecognizer(config)

        s1 = recognizer.create_stream()
        s2 = recognizer.create_stream()

        s1.accept_wave_file(wave1)
        s2.accept_wave_file(wave2)

        recognizer.decode_streams([s1, s2])
        print(s1.result)
        print(s2.result)

    def test_icefall_ctc_model_hlg_decoding(self):
        nn_model = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt"
        tokens = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/tokens.txt"
        hlg = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt"
        wave1 = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav"
        wave2 = f"{d}/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav"

        if not Path(nn_model).is_file():
            print("skipping test_icefall_ctc_model_hlg_decoding()")
            return
        print()
        print("test_icefall_ctc_model_hlg_decoding()")

        feat_config = sherpa.FeatureConfig()

        feat_config.fbank_opts.frame_opts.samp_freq = 16000
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        ctc_decoder_config = sherpa.OfflineCtcDecoderConfig(hlg=hlg)

        config = sherpa.OfflineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            feat_config=feat_config,
            ctc_decoder_config=ctc_decoder_config,
        )

        recognizer = sherpa.OfflineRecognizer(config)

        s1 = recognizer.create_stream()
        s2 = recognizer.create_stream()

        s1.accept_wave_file(wave1)
        s2.accept_wave_file(wave2)

        recognizer.decode_streams([s1, s2])
        print(s1.result)
        print(s2.result)

    def test_wenet_ctc_model(self):
        nn_model = f"{d}/wenet-english-model/final.zip"
        tokens = f"{d}/wenet-english-model/units.txt"
        wave1 = f"{d}/wenet-english-model/test_wavs/1089-134686-0001.wav"
        wave2 = f"{d}/wenet-english-model/test_wavs/1221-135766-0001.wav"

        if not Path(nn_model).is_file():
            print("skipping test_wenet_ctc_model()")
            return
        print()
        print("------test_wenet_ctc_model()------")

        # models from wenet expect un-normalized audio samples
        feat_config = sherpa.FeatureConfig(normalize_samples=False)

        feat_config.fbank_opts.frame_opts.samp_freq = 16000
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        config = sherpa.OfflineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
        )

        recognizer = sherpa.OfflineRecognizer(config)

        s1 = recognizer.create_stream()
        s2 = recognizer.create_stream()

        s1.accept_wave_file(wave1)
        s2.accept_wave_file(wave2)

        recognizer.decode_streams([s1, s2])
        print(s1.result)
        print(s2.result)

    def test_torchaudio_wav2vec2_0_ctc_model(self):
        nn_model = f"{d}/wav2vec2.0-torchaudio/wav2vec2_asr_base_960h.pt"
        tokens = f"{d}/wav2vec2.0-torchaudio/tokens.txt"
        wave1 = f"{d}/wav2vec2.0-torchaudio/test_wavs/1089-134686-0001.wav"
        wave2 = f"{d}/wav2vec2.0-torchaudio/test_wavs/1221-135766-0001.wav"

        if not Path(nn_model).is_file():
            print("skipping test_torchaudio_wav2vec2_0_ctc_model()")
            return

        print()
        print("test_torchaudio_wav2vec2_0_ctc_model()")

        config = sherpa.OfflineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
        )

        recognizer = sherpa.OfflineRecognizer(config)

        s1 = recognizer.create_stream()
        s2 = recognizer.create_stream()

        s1.accept_wave_file(wave1)
        s2.accept_wave_file(wave2)

        recognizer.decode_streams([s1, s2])
        print(s1.result)
        print(s2.result)

    def test_icefall_transducer_model(self):
        nn_model = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14/exp/cpu_jit.pt"
        tokens = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14/data/lang_bpe_500/tokens.txt"
        wave1 = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14/test_wavs/1089-134686-0001.wav"
        wave2 = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless8-2022-11-14/test_wavs/1221-135766-0001.wav"

        if not Path(nn_model).is_file():
            print("skipping test_icefall_transducer_model()")
            return

        print()
        print("test_icefall_transducer_model()")

        feat_config = sherpa.FeatureConfig()

        feat_config.fbank_opts.frame_opts.samp_freq = 16000
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        config = sherpa.OfflineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
        )

        recognizer = sherpa.OfflineRecognizer(config)

        s1 = recognizer.create_stream()
        s2 = recognizer.create_stream()

        s1.accept_wave_file(wave1)
        s2.accept_wave_file(wave2)

        recognizer.decode_streams([s1, s2])
        print(s1.result)
        print(s2.result)


if __name__ == "__main__":
    unittest.main()
