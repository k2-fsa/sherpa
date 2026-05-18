// Copyright (c)  2026  Xiaomi Corporation
//
// Asynchronous text-to-speech with the Supertonic model.
// Uses createAsync() and generateAsync() for non-blocking generation
// with a progress callback.
//
// Usage:
//   node offline_tts_async.js
//
const sherpa_onnx = require('sherpa-onnx-node');

async function createOfflineTts() {
  const config = {
    model: {
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
      debug: false,
      numThreads: 2,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };

  // createAsync() returns a Promise that resolves to an OfflineTts instance.
  return await sherpa_onnx.OfflineTts.createAsync(config);
}

async function generateAudioAsync(tts, text) {
  const generationConfig = new sherpa_onnx.GenerationConfig({
    sid: 6,
    speed: 1.25,
    numSteps: 8,
    extra: {lang: 'en'},
  });

  console.log('Starting generation...');

  // generateAsync() returns a Promise. The onProgress callback is invoked
  // with {samples, progress} after each chunk is generated.
  // Return a truthy value (e.g. 1) to continue, or a falsy value to cancel.
  const audio = await tts.generateAsync({
    text,
    enableExternalBuffer: true,
    generationConfig,
    onProgress: ({samples, progress}) => {
      process.stdout.write(
          `Progress: ${(progress * 100).toFixed(1)}%, ` +
          `Samples: ${samples.length}\r`);
      return 1;  // continue generation
    },
  });

  console.log('\nGeneration complete!');
  return audio;
}

async function main() {
  console.log('Creating OfflineTts...');
  const tts = await createOfflineTts();
  console.log('OfflineTts created!');

  const text =
      'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

  const start = Date.now();
  const audio = await generateAudioAsync(tts, text);
  const stop = Date.now();

  const elapsed_seconds = (stop - start) / 1000;
  const duration = audio.samples.length / audio.sampleRate;
  const real_time_factor = elapsed_seconds / duration;

  console.log('Wave duration', duration.toFixed(3), 'seconds');
  console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
      real_time_factor.toFixed(3));

  const filename = 'test-supertonic-en-async.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
});
