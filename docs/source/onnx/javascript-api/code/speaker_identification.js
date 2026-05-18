// Copyright (c)  2024  Xiaomi Corporation
//
// Speaker identification and verification using speaker embeddings.
//
// Usage:
//   node speaker_identification.js
//
const sherpa_onnx = require('sherpa-onnx-node');
const assert = require('node:assert');

// Download the embedding model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
function createSpeakerEmbeddingExtractor() {
  const config = {
    model: './3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx',
    numThreads: 1,
    debug: true,
  };
  return new sherpa_onnx.SpeakerEmbeddingExtractor(config);
}

// Helper: read a WAV file and compute its speaker embedding.
function computeEmbedding(extractor, filename) {
  const stream = extractor.createStream();
  const wave = sherpa_onnx.readWave(filename);
  stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});
  return extractor.compute(stream);
}

const extractor = createSpeakerEmbeddingExtractor();

// SpeakerEmbeddingManager stores speaker embeddings and supports
// search, verify, add, and remove operations.
const manager = new sherpa_onnx.SpeakerEmbeddingManager(extractor.dim);

// --- Enroll speakers ---
// Download test files from https://github.com/csukuangfj/sr-data
const spk1Files = [
  './sr-data/enroll/fangjun-sr-1.wav',
  './sr-data/enroll/fangjun-sr-2.wav',
  './sr-data/enroll/fangjun-sr-3.wav',
];

let spk1Vec = [];
for (let f of spk1Files) {
  spk1Vec.push(computeEmbedding(extractor, f));
}

const spk2Files = [
  './sr-data/enroll/leijun-sr-1.wav',
  './sr-data/enroll/leijun-sr-2.wav',
];

let spk2Vec = [];
for (let f of spk2Files) {
  spk2Vec.push(computeEmbedding(extractor, f));
}

// addMulti() registers a speaker with multiple enrollment utterances.
let ok = manager.addMulti({name: 'fangjun', v: spk1Vec});
assert.equal(ok, true);

ok = manager.addMulti({name: 'leijun', v: spk2Vec});
assert.equal(ok, true);

assert.equal(manager.getNumSpeakers(), 2);
assert.equal(manager.contains('fangjun'), true);
assert.equal(manager.contains('leijun'), true);

console.log('--- All speakers ---');
console.log(manager.getAllSpeakerNames());
console.log('--------------------');

// --- Identify test utterances ---
const testFiles = [
  './sr-data/test/fangjun-test-sr-1.wav',
  './sr-data/test/leijun-test-sr-1.wav',
  './sr-data/test/liudehua-test-sr-1.wav',
];

const threshold = 0.6;

for (let f of testFiles) {
  const embedding = computeEmbedding(extractor, f);

  // search() returns the speaker name, or '' if no match above threshold.
  let name = manager.search({v: embedding, threshold: threshold});
  if (name == '') {
    name = '<Unknown>';
  }
  console.log(`${f}: ${name}`);
}

// --- Verify a specific speaker ---
ok = manager.verify({
  name: 'fangjun',
  v: computeEmbedding(extractor, testFiles[0]),
  threshold: threshold
});
assert.equal(ok, true);

// --- Remove a speaker ---
ok = manager.remove('fangjun');
assert.equal(ok, true);

ok = manager.verify({
  name: 'fangjun',
  v: computeEmbedding(extractor, testFiles[0]),
  threshold: threshold
});
assert.equal(ok, false);

assert.equal(manager.getNumSpeakers(), 1);
