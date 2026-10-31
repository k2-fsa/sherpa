Models
======

We support the following ASR models from `Omnilingual ASR`_

  - ``omniASR_CTC_300M``
  - ``omniASR_CTC_1B``

You can find the download links below:

.. list-table::

 * - Model Name
   - | Download URL
     | (GitHub)
   - | Download URL
     | (Huggingface)
   - Comment
 * - omniASR_CTC_300M
   - `URL <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12.tar.bz2>`_
   - `URL <https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12/tree/main>`_
   - float32 weights
 * - omniASR_CTC_300M int8
   - `URL <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2>`_
   - `URL <https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tree/main>`_
   - int8 weights
 * - omniASR_CTC_1B
   - ``N/A``
   - `URL <https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-2025-11-12/tree/main>`_
   - float32 weights
 * - omniASR_CTC_1B int8
   - `URL <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-int8-2025-11-12.tar.bz2>`_
   - `URL <https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-int8-2025-11-12/tree/main>`_
   - int8 weights


sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12 (1600+ languages)
-------------------------------------------------------------------------------------

In the following we show how to use ``omniASR_CTC_300M int8``.

.. hint::

   Usage for other models is similar to this one.

Download the model
::::::::::::::::::

Please use the following code to download the model:

.. code-block::

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
   tar xvf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
   rm sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2

   ls -lh sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12

You should see the following output::

  ls -lh sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/
  total 713792
  -rw-r--r--@ 1 fangjun  staff   581B 12 Nov 20:19 LICENSE
  -rw-r--r--@ 1 fangjun  staff   348M 12 Nov 20:14 model.int8.onnx
  -rw-r--r--@ 1 fangjun  staff    11K 12 Nov 20:19 README.md
  drwxr-xr-x@ 7 fangjun  staff   224B 13 Nov 14:41 test_wavs
  -rw-r--r--@ 1 fangjun  staff    84K 12 Nov 20:19 tokens.txt

Decode wave files
^^^^^^^^^^^^^^^^^

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --omnilingual-asr-model=./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/model.int8.onnx \
    --tokens=./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt \
    ./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/test_wavs/en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code/int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --omnilingual-asr-model=./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/model.int8.onnx \
    --tokens=./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --omnilingual-asr-model=./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/model.int8.onnx \
    --tokens=./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt
