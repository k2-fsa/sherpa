// Copyright (c)  2025  Xiaomi Corporation
//
// Text-to-speech with the Kitten Nano model (async generation).
//
// Usage:
//   node tts_kitten_async.js
//
const sherpa_onnx = require('sherpa-onnx-node');

async function createOfflineTts() {
  const config = {
    model: {
      kitten: {
        model: './kitten-nano-en-v0_1-fp16/model.fp16.onnx',
        voices: './kitten-nano-en-v0_1-fp16/voices.bin',
        tokens: './kitten-nano-en-v0_1-fp16/tokens.txt',
        dataDir: './kitten-nano-en-v0_1-fp16/espeak-ng-data',
      },
      debug: true,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };

  return await sherpa_onnx.OfflineTts.createAsync(config);
}

async function main() {
  const tts = await createOfflineTts();

  const text =
      'Today as always, men fall into two groups: slaves and free men. ' +
      'Whoever does not have two-thirds of his day for himself, is a slave, ' +
      'whatever he may be: a statesman, a businessman, an official, or a scholar.';

  console.log('Number of speakers:', tts.numSpeakers);
  console.log('Sample rate:', tts.sampleRate);

  const start = Date.now();
  const generationConfig = new sherpa_onnx.GenerationConfig({
    sid: 6,
    speed: 1.0,
    silenceScale: 0.2,
  });

  const audio = await tts.generateAsync({
    text,
    generationConfig,
    onProgress({samples, progress}) {
      process.stdout.write(`\rGenerating... ${
          (progress * 100).toFixed(1)}% (chunk length: ${samples.length})`);
      return true;
    },
  });

  console.log('\nGeneration finished.');

  const stop = Date.now();
  const elapsedSeconds = (stop - start) / 1000;
  const durationSeconds = audio.samples.length / audio.sampleRate;
  const realTimeFactor = elapsedSeconds / durationSeconds;

  console.log('Wave duration:', durationSeconds.toFixed(3), 'seconds');
  console.log('Elapsed time:', elapsedSeconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${elapsedSeconds.toFixed(3)} / ${durationSeconds.toFixed(3)} =`,
      realTimeFactor.toFixed(3));

  const filename = 'test-kitten-en.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });

  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('TTS failed:', err);
  process.exit(1);
});
