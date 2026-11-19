.. _onnx-x-asr-javascript-wasm-api:

JavaScript (WASM)
=================

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the WASM version of `sherpa-onnx`_ in Node.js. Everything runs inside the WebAssembly virtual machine; no platform-specific native library is needed. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- Node.js 18 or above

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Install the package
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   npm install sherpa-onnx

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``x-asr-offline.js``:

.. code-block:: javascript

   // This file shows how to use a non-streaming X-ASR model with the WASM
   // version of sherpa-onnx in Node.js.
   //
   // Before you run it, please
   //
   //   npm install sherpa-onnx
   //
   // and download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   const sherpa_onnx = require('sherpa-onnx');

   const modelDir =
       './sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03';

   // Step 1: Create the recognizer from the X-ASR model files.
   const config = {
     modelConfig: {
       transducer: {
         encoder: `${modelDir}/encoder-epoch-99-avg-1.int8.onnx`,
         decoder: `${modelDir}/decoder-epoch-99-avg-1.onnx`,
         joiner: `${modelDir}/joiner-epoch-99-avg-1.int8.onnx`,
       },
       tokens: `${modelDir}/tokens.txt`,
       numThreads: 2,
     },
   };
   const recognizer = sherpa_onnx.createOfflineRecognizer(config);

   // Step 2: Read the wave file you want to transcribe.
   const filename = `${modelDir}/test_wavs/0.wav`;
   const wave = sherpa_onnx.readWave(filename);

   // Step 3: Create a stream, send the whole audio to it, and decode.
   const start = Date.now();
   const stream = recognizer.createStream();
   stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});
   recognizer.decode(stream);
   const elapsed = (Date.now() - start) / 1000;

   // Step 4: Print the result. X-ASR punct models also add punctuations.
   console.log(filename);
   console.log(recognizer.getResult(stream).text);

   const audioDuration = wave.samples.length / wave.sampleRate;
   console.log(`audio duration: ${audioDuration.toFixed(3)} s`);
   console.log(`elapsed seconds: ${elapsed.toFixed(3)} s`);
   console.log(
       `RTF = ${elapsed.toFixed(3)}/${audioDuration.toFixed(3)} = ${(elapsed / audioDuration).toFixed(3)}`);

   stream.free();
   recognizer.free();

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree -I node_modules .
   .
   ├── package-lock.json
   ├── package.json
   ├── sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   │   ├── bpe.model
   │   ├── decoder-epoch-99-avg-1.onnx
   │   ├── encoder-epoch-99-avg-1.int8.onnx
   │   ├── joiner-epoch-99-avg-1.int8.onnx
   │   ├── README.md
   │   ├── test_onnx.py
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   └── 3.wav
   │   └── tokens.txt
   └── x-asr-offline.js

   3 directories, 14 files

   $ ls -lh
   total 24
   drwxr-xr-x@  4 fangjun  wheel   128B 19 Nov 12:41 node_modules
   -rw-r--r--@  1 fangjun  wheel   478B 19 Nov 12:41 package-lock.json
   -rw-r--r--@  1 fangjun  wheel    56B 19 Nov 12:41 package.json
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   -rw-r--r--@  1 fangjun  wheel   1.9K 19 Nov 12:41 x-asr-offline.js

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   node ./x-asr-offline.js

Output
^^^^^^

.. code-block:: text

   node:internal/modules/cjs/loader:1424
     throw err;
     ^

   Error: Cannot find module './sherpa-onnx-wasm-nodejs.js'
   Require stack:
   - /private/tmp/java-doc-work/x-asr-demos/node-wasm-offline.js
       at Module._resolveFilename (node:internal/modules/cjs/loader:1421:15)
       at defaultResolveImpl (node:internal/modules/cjs/loader:1059:19)
       at resolveForCJSWithHooks (node:internal/modules/cjs/loader:1064:22)
       at Module._load (node:internal/modules/cjs/loader:1227:37)
       at TracingChannel.traceSync (node:diagnostics_channel:328:14)
       at wrapModuleLoad (node:internal/modules/cjs/loader:245:24)
       at Module.require (node:internal/modules/cjs/loader:1504:12)
       at require (node:internal/modules/helpers:152:16)
       at Module._compile (node:internal/modules/cjs/loader:1761:14) {
     code: 'MODULE_NOT_FOUND',
     requireStack: [
       '/private/tmp/java-doc-work/x-asr-demos/node-wasm-offline.js'
     ]
   }

   Node.js v24.11.1

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Streaming X-ASR
---------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Install the package
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   npm install sherpa-onnx

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``x-asr-streaming.js``:

.. code-block:: javascript

   // This file shows how to use a streaming X-ASR model with the WASM
   // version of sherpa-onnx in Node.js.
   //
   // Before you run it, please
   //
   //   npm install sherpa-onnx
   //
   // and download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   const sherpa_onnx = require('sherpa-onnx');

   const modelDir =
       './sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05';

   // Step 1: Create the recognizer from the X-ASR model files.
   const config = {
     modelConfig: {
       transducer: {
         encoder: `${modelDir}/encoder.int8.onnx`,
         decoder: `${modelDir}/decoder.onnx`,
         joiner: `${modelDir}/joiner.int8.onnx`,
       },
       tokens: `${modelDir}/tokens.txt`,
       numThreads: 2,
     },
   };
   const recognizer = sherpa_onnx.createOnlineRecognizer(config);

   // Step 2: Read the wave file you want to transcribe.
   const filename = `${modelDir}/test_wavs/0.wav`;
   const wave = sherpa_onnx.readWave(filename);

   // Step 3: Create a stream and feed the audio in small chunks, just like
   // a microphone would do in a real-time application.
   const start = Date.now();
   const stream = recognizer.createStream();

   // 0.2 second for 16 kHz audio
   const chunk = 3200;

   let last = '';
   for (let i = 0; i < wave.samples.length; i += chunk) {
     stream.acceptWaveform({
       sampleRate: wave.sampleRate,
       samples: wave.samples.subarray(i, Math.min(i + chunk, wave.samples.length)),
     });

     while (recognizer.isReady(stream)) {
       recognizer.decode(stream);
     }

     // Print the latest partial result when it changes.
     const text = recognizer.getResult(stream).text;
     if (text.length > 0 && text !== last) {
       last = text;
       console.log(text);
     }
   }

   // Send some tail padding so the last words are flushed out.
   const tailPadding = new Float32Array(wave.sampleRate * 0.5);
   stream.acceptWaveform({samples: tailPadding, sampleRate: wave.sampleRate});
   stream.inputFinished();
   while (recognizer.isReady(stream)) {
     recognizer.decode(stream);
   }

   // Step 4: Print the final result.
   const elapsed = (Date.now() - start) / 1000;
   console.log('Final result:', recognizer.getResult(stream).text);

   const audioDuration = wave.samples.length / wave.sampleRate;
   console.log(`audio duration: ${audioDuration.toFixed(3)} s`);
   console.log(`elapsed seconds: ${elapsed.toFixed(3)} s`);
   console.log(
       `RTF = ${elapsed.toFixed(3)}/${audioDuration.toFixed(3)} = ${(elapsed / audioDuration).toFixed(3)}`);

   stream.free();
   recognizer.free();

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree -I node_modules .
   .
   ├── package-lock.json
   ├── package.json
   ├── sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   │   ├── bpe.model
   │   ├── decoder.onnx
   │   ├── encoder.int8.onnx
   │   ├── joiner.int8.onnx
   │   ├── README.md
   │   ├── test_onnx.py
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   └── 3.wav
   │   └── tokens.txt
   └── x-asr-streaming.js

   3 directories, 14 files

   $ ls -lh
   total 24
   drwxr-xr-x@  4 fangjun  wheel   128B 19 Nov 12:41 node_modules
   -rw-r--r--@  1 fangjun  wheel   480B 19 Nov 12:41 package-lock.json
   -rw-r--r--@  1 fangjun  wheel    56B 19 Nov 12:41 package.json
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   -rw-r--r--@  1 fangjun  wheel   2.6K 19 Nov 12:41 x-asr-streaming.js

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   node ./x-asr-streaming.js

Output
^^^^^^

.. code-block:: text

   node:internal/modules/cjs/loader:1424
     throw err;
     ^

   Error: Cannot find module './sherpa-onnx-wasm-nodejs.js'
   Require stack:
   - /private/tmp/java-doc-work/x-asr-demos/node-wasm-streaming.js
       at Module._resolveFilename (node:internal/modules/cjs/loader:1421:15)
       at defaultResolveImpl (node:internal/modules/cjs/loader:1059:19)
       at resolveForCJSWithHooks (node:internal/modules/cjs/loader:1064:22)
       at Module._load (node:internal/modules/cjs/loader:1227:37)
       at TracingChannel.traceSync (node:diagnostics_channel:328:14)
       at wrapModuleLoad (node:internal/modules/cjs/loader:245:24)
       at Module.require (node:internal/modules/cjs/loader:1504:12)
       at require (node:internal/modules/helpers:152:16)
       at Module._compile (node:internal/modules/cjs/loader:1761:14) {
     code: 'MODULE_NOT_FOUND',
     requireStack: [
       '/private/tmp/java-doc-work/x-asr-demos/node-wasm-streaming.js'
     ]
   }

   Node.js v24.11.1

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Notes
-----

The WASM package is ``sherpa-onnx``; the native addon used on the :ref:`onnx-x-asr-javascript-api` page is ``sherpa-onnx-node``. The APIs are almost the same.

