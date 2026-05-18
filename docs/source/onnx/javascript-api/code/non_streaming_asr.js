// Copyright (c)  2024  Xiaomi Corporation
//
// Non-streaming (offline) automatic speech recognition with a Zipformer
// transducer model.
//
// Usage:
//   node non_streaming_asr.js
//
const sherpa_onnx = require('sherpa-onnx-node');

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'transducer': {
      'encoder':
          './sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.int8.onnx',
      'decoder':
          './sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx',
      'joiner':
          './sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.int8.onnx',
    },
    'tokens': './sherpa-onnx-zipformer-en-2023-04-01/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
  }
};

const waveFilename = './sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav';

// Create the recognizer and a stream.
const recognizer = new sherpa_onnx.OfflineRecognizer(config);
const stream = recognizer.createStream();

// Read the wave file and feed it to the stream.
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

// Decode and get the result.
let start = Date.now();
recognizer.decode(stream);
const result = recognizer.getResult(stream);
let stop = Date.now();

const elapsed_seconds = (stop - start) / 1000;
const duration = wave.samples.length / wave.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds');
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3));
console.log(waveFilename);
console.log('result\n', result);
