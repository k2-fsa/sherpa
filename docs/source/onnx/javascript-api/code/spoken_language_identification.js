// Copyright (c)  2023-2024  Xiaomi Corporation
//
// Spoken language identification using a Whisper multilingual model.
//
// Usage:
//   node spoken_language_identification.js
//
const sherpa_onnx = require('sherpa-onnx-node');

// Download whisper multi-lingual models from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
function createSpokenLanguageID() {
  const config = {
    whisper: {
      encoder: './sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx',
      decoder: './sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx',
    },
    debug: true,
    numThreads: 1,
    provider: 'cpu',
  };
  return new sherpa_onnx.SpokenLanguageIdentification(config);
}

const slid = createSpokenLanguageID();

const testWaves = [
  './spoken-language-identification-test-wavs/ar-arabic.wav',
  './spoken-language-identification-test-wavs/de-german.wav',
  './spoken-language-identification-test-wavs/en-english.wav',
  './spoken-language-identification-test-wavs/fr-french.wav',
  './spoken-language-identification-test-wavs/pt-portuguese.wav',
  './spoken-language-identification-test-wavs/es-spanish.wav',
  './spoken-language-identification-test-wavs/zh-chinese.wav',
];

// Intl.DisplayNames converts ISO language codes to human-readable names.
const display = new Intl.DisplayNames(['en'], {type: 'language'});

for (let f of testWaves) {
  const stream = slid.createStream();

  const wave = sherpa_onnx.readWave(f);
  stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

  const lang = slid.compute(stream);
  console.log(`${f}: ${lang} (${display.of(lang)})`);
}
