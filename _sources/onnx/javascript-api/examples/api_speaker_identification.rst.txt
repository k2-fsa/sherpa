Speaker Identification API
==========================

Speaker identification and verification API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/speaker-identification.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/speaker-identification.js>`_

API
---

SpeakerEmbeddingExtractor
^^^^^^^^^^^^^^^^^^^^^^^^^

Extracts speaker embedding vectors from audio.

Constructor
"""""""""""

.. code-block:: javascript

   const extractor = new sherpa_onnx.SpeakerEmbeddingExtractor(config);

:param config: Configuration object with:

- ``model`` (string, optional) — Path to the embedding ONNX model.
- ``numThreads`` (number, optional).
- ``debug`` (boolean, optional).
- ``provider`` (string, optional).

Methods
"""""""

``extractor.createStream()``
.............................

:returns: A new ``OnlineStream`` for feeding audio.

``extractor.isReady(stream)``
..............................

Check if the stream has enough samples for embedding computation.

:param stream: An ``OnlineStream``.
:returns: ``true`` if ready (``boolean``).

``extractor.compute(stream, enableExternalBuffer?)``
.....................................................

Compute the speaker embedding.

:param stream: An ``OnlineStream``.
:param enableExternalBuffer: Whether to use an external buffer (``boolean``, default ``true``).
:returns: Embedding vector (``Float32Array``).

Properties
""""""""""

- ``extractor.config`` — The configuration object.
- ``extractor.dim`` — Embedding dimension (number).

SpeakerEmbeddingManager
^^^^^^^^^^^^^^^^^^^^^^^^

Manages a collection of speaker embeddings for identification and verification.

Constructor
"""""""""""

.. code-block:: javascript

   const manager = new sherpa_onnx.SpeakerEmbeddingManager(dim);

:param dim: Embedding dimension (number). Must match the extractor's ``dim``.

Methods
"""""""

``manager.add(obj)``
.....................

Register a speaker with a single embedding.

:param obj: ``{ name: string, v: Float32Array }``.
:returns: ``true`` on success (``boolean``).

``manager.addMulti(obj)``
...........................

Register a speaker with multiple embeddings.

:param obj: ``{ name: string, v: Float32Array[] }``.
:returns: ``true`` on success (``boolean``).

``manager.remove(name)``
.........................

Remove a speaker by name.

:param name: Speaker name (string).
:returns: ``true`` if removed (``boolean``).

``manager.search(obj)``
........................

Find the speaker matching an embedding.

:param obj: ``{ v: Float32Array, threshold: number }``.
:returns: Speaker name (string), or empty string if no match.

``manager.verify(obj)``
........................

Verify if an embedding matches a specific speaker.

:param obj: ``{ name: string, v: Float32Array, threshold: number }``.
:returns: ``true`` if verified (``boolean``).

``manager.contains(name)``
...........................

Check if a speaker exists.

:param name: Speaker name (string).
:returns: ``true`` if exists (``boolean``).

``manager.getNumSpeakers()``
.............................

:returns: Number of registered speakers (``number``).

``manager.getAllSpeakerNames()``
.................................

:returns: Array of all speaker names (``string[]``).

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   const extractor = new sherpa_onnx.SpeakerEmbeddingExtractor({
     model: './3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx'
   });
   const manager = new sherpa_onnx.SpeakerEmbeddingManager(extractor.dim);

   // Register a speaker
   const stream = extractor.createStream();
   stream.acceptWaveform({ samples: enrollSamples, sampleRate: 16000 });
   const embedding = extractor.compute(stream);
   manager.add({ name: 'alice', v: embedding });

   // Identify a speaker
   const stream2 = extractor.createStream();
   stream2.acceptWaveform({ samples: testSamples, sampleRate: 16000 });
   const embedding2 = extractor.compute(stream2);
   const name = manager.search({ v: embedding2, threshold: 0.5 });
   console.log(`Identified: ${name}`);

Notes
-----

- The embedding dimension must match between the extractor and manager.
- Use ``addMulti()`` to register multiple enrollment recordings for better accuracy.
- The ``threshold`` controls the trade-off between false accepts and false rejects.
