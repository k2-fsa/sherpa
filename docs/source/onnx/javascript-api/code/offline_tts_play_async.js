// Copyright (c)  2026  Xiaomi Corporation
//
// Asynchronous text-to-speech with real-time playback using the speaker
// npm package. Audio chunks are played as they are generated.
//
// Usage:
//   npm install speaker
//   node offline_tts_play_async.js
//
const Speaker = require('speaker');
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

  return await sherpa_onnx.OfflineTts.createAsync(config);
}

function createSpeaker(sampleRate) {
  return new Speaker({
    channels: 1,
    bitDepth: 16,
    sampleRate: sampleRate,
    signed: true,
  });
}

// Convert Float32 samples [-1.0, 1.0] to Int16 buffer for the speaker.
function float32ToInt16Buffer(samples) {
  const buffer = Buffer.alloc(samples.length * 2);

  for (let i = 0; i < samples.length; ++i) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    const v = s < 0 ? s * 0x8000 : s * 0x7fff;
    buffer.writeInt16LE(Math.round(v), i * 2);
  }

  return buffer;
}

function waitForEvent(emitter, eventName) {
  return new Promise((resolve, reject) => {
    emitter.once(eventName, resolve);
    emitter.once('error', reject);
  });
}

async function generateAudioAsync(tts, text) {
  const generationConfig = new sherpa_onnx.GenerationConfig({
    sid: 6,
    speed: 1.25,
    numSteps: 8,
    extra: {lang: 'en'},
  });

  const speaker = createSpeaker(tts.sampleRate);
  const start = Date.now();

  console.log('Starting generation and playback...');

  // Each onProgress callback receives a chunk of generated audio.
  // We convert it to Int16 and pipe it to the speaker for immediate playback.
  const audio = await tts.generateAsync({
    text,
    enableExternalBuffer: true,
    generationConfig,
    onProgress: ({samples, progress}) => {
      process.stdout.write(
          `Progress: ${(progress * 100).toFixed(1)}%, ` +
          `Chunk samples: ${samples.length}\r`);
      speaker.write(float32ToInt16Buffer(samples));
      return 1;
    },
  });

  const generationStop = Date.now();
  speaker.end();
  await waitForEvent(speaker, 'close');
  const playbackStop = Date.now();

  console.log('\nGeneration and playback complete!');
  return {
    audio,
    generationElapsedSeconds: (generationStop - start) / 1000,
    playbackElapsedSeconds: (playbackStop - start) / 1000,
  };
}

async function main() {
  console.log('Creating OfflineTts...');
  const tts = await createOfflineTts();
  console.log('OfflineTts created!');

  const text =
      'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

  const {audio, generationElapsedSeconds, playbackElapsedSeconds} =
      await generateAudioAsync(tts, text);
  const duration = audio.samples.length / audio.sampleRate;
  const real_time_factor = generationElapsedSeconds / duration;

  console.log('Wave duration', duration.toFixed(3), 'seconds');
  console.log(
      'Generation elapsed', generationElapsedSeconds.toFixed(3), 'seconds');
  console.log(
      'Playback drained in', playbackElapsedSeconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${generationElapsedSeconds.toFixed(3)}/${duration.toFixed(3)} =`,
      real_time_factor.toFixed(3));

  const filename = 'test-supertonic-en-play-async.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
});
