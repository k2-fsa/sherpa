// Copyright (c)  2024  Xiaomi Corporation
//
// Audio tagging: classify audio events in WAV files using a CED model.
//
// Usage:
//   node audio_tagging.js
//
const sherpa_onnx = require('sherpa-onnx-node');

// Download models from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
function createAudioTagging() {
  const config = {
    model: {
      ced: './sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.int8.onnx',
      numThreads: 1,
      debug: true,
    },
    labels:
        './sherpa-onnx-ced-mini-audio-tagging-2024-04-19/class_labels_indices.csv',
    topK: 5,  // return the top-5 most probable audio events
  };
  return new sherpa_onnx.AudioTagging(config);
}

const at = createAudioTagging();

const testWaves = [
  './sherpa-onnx-ced-mini-audio-tagging-2024-04-19/test_wavs/1.wav',
  './sherpa-onnx-ced-mini-audio-tagging-2024-04-19/test_wavs/2.wav',
  './sherpa-onnx-ced-mini-audio-tagging-2024-04-19/test_wavs/3.wav',
];

console.log('------');

for (let filename of testWaves) {
  const start = Date.now();
  const stream = at.createStream();
  const wave = sherpa_onnx.readWave(filename);
  stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

  // compute() returns an array of {prob, name} objects.
  const events = at.compute(stream);
  const stop = Date.now();

  const elapsed_seconds = (stop - start) / 1000;
  const duration = wave.samples.length / wave.sampleRate;
  const real_time_factor = elapsed_seconds / duration;

  console.log('input file:', filename);
  console.log('Probability\t\tName');
  for (let e of events) {
    console.log(`${e.prob.toFixed(3)}\t\t\t${e.name}`);
  }
  console.log('Wave duration', duration.toFixed(3), 'seconds');
  console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
      real_time_factor.toFixed(3));
  console.log('------');
}
