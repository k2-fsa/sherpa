/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "sherpa-onnx C API", "index.html", [
    [ "sherpa-onnx public API documentation", "index.html", "index" ],
    [ "Non-Streaming (Offline) ASR Models", "offline_asr.html", [
      [ "Zipformer Transducer", "offline_asr.html#offline_asr_zipformer_transducer", null ],
      [ "Zipformer CTC", "offline_asr.html#offline_asr_zipformer_ctc", null ],
      [ "Whisper", "offline_asr.html#offline_asr_whisper", null ],
      [ "SenseVoice", "offline_asr.html#offline_asr_sense_voice", null ],
      [ "NeMo Parakeet TDT", "offline_asr.html#offline_asr_parakeet", null ],
      [ "GigaAM v2 (NeMo Transducer, Russian)", "offline_asr.html#offline_asr_giga_am", null ],
      [ "NeMo CTC", "offline_asr.html#offline_asr_nemo_ctc", null ],
      [ "Paraformer", "offline_asr.html#offline_asr_paraformer", null ],
      [ "Moonshine", "offline_asr.html#offline_asr_moonshine", null ],
      [ "FireRedAsr", "offline_asr.html#offline_asr_fire_red", null ],
      [ "FireRedAsr CTC", "offline_asr.html#offline_asr_fire_red_ctc", null ],
      [ "Dolphin", "offline_asr.html#offline_asr_dolphin", null ],
      [ "NeMo Canary", "offline_asr.html#offline_asr_canary", null ],
      [ "Cohere Transcribe", "offline_asr.html#offline_asr_cohere", null ],
      [ "WeNet CTC", "offline_asr.html#offline_asr_wenet", null ],
      [ "Omnilingual", "offline_asr.html#offline_asr_omnilingual", null ],
      [ "FunASR Nano", "offline_asr.html#offline_asr_funasr", null ],
      [ "Qwen3-ASR", "offline_asr.html#offline_asr_qwen3", null ],
      [ "MedASR", "offline_asr.html#offline_asr_medasr", null ],
      [ "TeleSpeech CTC", "offline_asr.html#offline_asr_telespeech", null ]
    ] ],
    [ "Streaming (Online) ASR Models", "online_asr.html", [
      [ "Transducer (Zipformer)", "online_asr.html#online_asr_transducer", null ],
      [ "Nemotron (NeMo Transducer)", "online_asr.html#online_asr_nemotron", null ],
      [ "Streaming Paraformer", "online_asr.html#online_asr_paraformer", null ],
      [ "Zipformer2 CTC", "online_asr.html#online_asr_zipformer2_ctc", null ],
      [ "T-One CTC", "online_asr.html#online_asr_t_one", null ]
    ] ],
    [ "Text-to-Speech (TTS) Models", "tts.html", [
      [ "Kokoro", "tts.html#tts_kokoro", null ],
      [ "VITS (Piper)", "tts.html#tts_vits", null ],
      [ "Matcha", "tts.html#tts_matcha", null ],
      [ "Kitten", "tts.html#tts_kitten", null ],
      [ "ZipVoice", "tts.html#tts_zipvoice", null ],
      [ "Pocket", "tts.html#tts_pocket", null ],
      [ "Supertonic", "tts.html#tts_supertonic", null ]
    ] ],
    [ "Voice Activity Detection (VAD)", "vad.html", [
      [ "Silero VAD", "vad.html#vad_silero", null ],
      [ "Ten VAD", "vad.html#vad_ten", null ]
    ] ],
    [ "Audio Tagging", "audio_tagging.html", [
      [ "Zipformer", "audio_tagging.html#audio_tagging_zipformer", null ],
      [ "CED", "audio_tagging.html#audio_tagging_ced", null ]
    ] ],
    [ "Punctuation Restoration", "punctuation.html", [
      [ "Offline Punctuation", "punctuation.html#punct_offline", null ],
      [ "Online Punctuation", "punctuation.html#punct_online", null ]
    ] ],
    [ "Speech Enhancement / Denoising", "speech_enhancement.html", [
      [ "Offline GTCRN", "speech_enhancement.html#se_offline_gtcrn", null ],
      [ "Offline DPDFNet", "speech_enhancement.html#se_offline_dpdfnet", null ],
      [ "Online GTCRN", "speech_enhancement.html#se_online_gtcrn", null ],
      [ "Online DPDFNet", "speech_enhancement.html#se_online_dpdfnet", null ]
    ] ],
    [ "Source Separation", "source_separation.html", [
      [ "Spleeter", "source_separation.html#ss_spleeter", null ],
      [ "UVR (MDX-Net)", "source_separation.html#ss_uvr", null ]
    ] ],
    [ "Offline Speaker Diarization", "speaker_diarization.html", null ],
    [ "Speaker Embedding Extraction and Management", "speaker_embedding.html", null ],
    [ "Spoken Language Identification", "spoken_language_id.html", null ],
    [ "Keyword Spotting", "keyword_spotting.html", null ],
    [ "Linear Resampler", "resampler.html", null ],
    [ "Deprecated List", "deprecated.html", null ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", "functions_vars" ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Functions", "globals_func.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"annotated.html",
"c-api_8h.html#ad85805b1b36228c8e2ba7b8d6e619749",
"classsherpa__onnx_1_1cxx_1_1OnlineStream.html#a02092d7c0325b17fb82b827979f20c88",
"structSherpaOnnxDenoisedAudio.html",
"structSherpaOnnxOfflineTtsMatchaModelConfig.html#a272d5b05976aa1fc5e91e3dc07597814",
"structsherpa__onnx_1_1cxx_1_1GenerationConfig.html",
"structsherpa__onnx_1_1cxx_1_1OfflineRecognizerConfig.html#a0aa817441873ab621d306ab26ffdf5bf",
"structsherpa__onnx_1_1cxx_1_1OfflineTtsSupertonicModelConfig.html#aac8b55566ec3da25707572eeabe534e9",
"structsherpa__onnx_1_1cxx_1_1SpokenLanguageIdentificationConfig.html#a3cf154efc354eeb7fda4e0905f942420"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';