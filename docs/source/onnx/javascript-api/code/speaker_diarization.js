// Copyright (c)  2024  Xiaomi Corporation
//
// Offline speaker diarization: determine who speaks when in an audio file.
//
// Usage:
//   node speaker_diarization.js
//
const sherpa_onnx = require('sherpa-onnx-node');

// Model files required:
// 1. Segmentation model:
//    https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
//
// 2. Embedding model:
//    https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
//
// 3. Test wave file:
//    https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

const config = {
  segmentation: {
    pyannote: {
      model: './sherpa-onnx-pyannote-segmentation-3-0/model.onnx',
    },
  },
  embedding: {
    model: './3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx',
  },
  clustering: {
    // Set numClusters to the expected number of speakers, or -1 to
    // let the algorithm decide automatically using the threshold.
    numClusters: 4,
    // A larger threshold leads to fewer clusters (fewer speakers).
    // Ignored when numClusters is not -1.
    threshold: 0.5,
  },
  // Discard segments shorter than minDurationOn seconds.
  minDurationOn: 0.2,
  // Merge two segments if the gap between them is less than minDurationOff.
  minDurationOff: 0.5,
};

const waveFilename = './0-four-speakers-zh.wav';

const sd = new sherpa_onnx.OfflineSpeakerDiarization(config);

const wave = sherpa_onnx.readWave(waveFilename);
if (sd.sampleRate != wave.sampleRate) {
  throw new Error(
      `Expected sample rate: ${sd.sampleRate}, given: ${wave.sampleRate}`);
}

const segments = sd.process(wave.samples);

// Each segment has: {start, end, speaker}
console.log('Segments:');
for (const seg of segments) {
  console.log(
      `  Speaker ${seg.speaker}: ${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s`);
}
