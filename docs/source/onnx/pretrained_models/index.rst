.. _sherpa-onnx-pre-trained-models:

Pre-trained models
==================

The following table lists links for all pre-trained models.


.. list-table::

 * - Description
   - URL
 * - Speech recognition (speech to text, ASR)
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_
 * - Text to speech (TTS)
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_
 * - VAD
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx>`_
 * - Keyword spotting
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models>`_
 * - Speech identification (Speaker ID)
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models>`_
 * - Spoken language identification (Language ID)
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_ (multi-lingual whisper)
 * - Audio tagging
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models>`_
 * - Punctuation
   - `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models>`_


In this section, we describe how to download and use all
available pre-trained models for speech recognition.


.. hint::

  Please install `git-lfs <https://git-lfs.com/>`_ before you continue.

  Otherwise, you will be ``SAD`` later.

.. toctree::
   :maxdepth: 5

   online-transducer/index
   online-paraformer/index
   online-ctc/index
   offline-transducer/index
   offline-paraformer/index
   offline-ctc/index
   telespeech/index
   whisper/index
   wenet/index
   small-online-models
