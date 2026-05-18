// Copyright (c)  2025  Xiaomi Corporation
//
// Offline speech enhancement (denoising) using a GTCRN model.
//
// Usage:
//   node speech_enhancement.js
//
const sherpa_onnx = require('sherpa-onnx-node');

// Download models from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
function createOfflineSpeechDenoiser() {
  const config = {
    model: {
      gtcrn: {model: './gtcrn_simple.onnx'},
      debug: true,
      numThreads: 1,
    },
  };
  return new sherpa_onnx.OfflineSpeechDenoiser(config);
}

const sd = createOfflineSpeechDenoiser();

const waveFilename = './inp_16k.wav';
const wave = sherpa_onnx.readWave(waveFilename);

// run() accepts {samples, sampleRate, enableExternalBuffer} and returns
// {samples, sampleRate}.
const denoised = sd.run({
  samples: wave.samples,
  sampleRate: wave.sampleRate,
  enableExternalBuffer: true
});

sherpa_onnx.writeWave(
    './enhanced-16k.wav',
    {samples: denoised.samples, sampleRate: denoised.sampleRate});

console.log(`Saved to ./enhanced-16k.wav`);
