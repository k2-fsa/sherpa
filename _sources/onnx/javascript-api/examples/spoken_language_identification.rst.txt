Spoken Language Identification
===============================

Identify the language spoken in a WAV file using a Whisper multilingual model.
This example classifies audio files into their corresponding languages.

Source file
-----------

`nodejs-addon-examples/test_spoken_language_identification.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_spoken_language_identification.js>`_

Code
----

.. literalinclude:: ../code/spoken_language_identification.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the Whisper multilingual model and test files::

     curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
     tar xvf sherpa-onnx-whisper-tiny.tar.bz2
     rm sherpa-onnx-whisper-tiny.tar.bz2

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
     tar xvf spoken-language-identification-test-wavs.tar.bz2
     rm spoken-language-identification-test-wavs.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node spoken_language_identification.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   ar-arabic.wav: ar (Arabic)
   de-german.wav: de (German)
   en-english.wav: en (English)
   fr-french.wav: fr (French)
   pt-portuguese.wav: pt (Portuguese)
   es-spanish.wav: es (Spanish)
   zh-chinese.wav: zh (Chinese)

Notes
-----

- ``SpokenLanguageIdentification`` requires a Whisper multilingual model
  (not an English-only model).
- ``compute()`` returns an ISO 639-1 language code (e.g., ``en``, ``zh``,
  ``fr``).
- ``Intl.DisplayNames`` is a built-in JavaScript API that converts language
  codes to human-readable names.
