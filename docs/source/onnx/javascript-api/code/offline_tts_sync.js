// Copyright (c)  2026  Xiaomi Corporation
//
// Synchronous text-to-speech with the Supertonic model.
//
// Usage:
//   node offline_tts_sync.js
//
const sherpa_onnx = require('sherpa-onnx-node');

function createOfflineTts() {
  const config = {
    model: {
      // Replace the paths below with the actual paths to your model files.
      supertonic: {
        durationPredictor:
            './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx',
        textEncoder:
            './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx',
        vectorEstimator:
            './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx',
        vocoder:
            './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx',
        ttsJson: './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json',
        unicodeIndexer:
            './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin',
        voiceStyle: './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin',
      },
      debug: true,
      numThreads: 2,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };
  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();

const text =
    'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

// GenerationConfig controls speaker ID, speed, number of diffusion steps,
// and language. The `extra.lang` field uses ISO 639-1 codes.
const generationConfig = new sherpa_onnx.GenerationConfig({
  sid: 6,           // speaker ID, valid range [0, 9]
  speed: 1.25,      // speech speed, 1.0 is normal
  numSteps: 8,      // number of diffusion steps
  extra: {lang: 'en'},  // language code
});

let start = Date.now();
const audio = tts.generate({text, generationConfig});
let stop = Date.now();

const elapsed_seconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds');
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3));

const filename = 'test-supertonic-en.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
