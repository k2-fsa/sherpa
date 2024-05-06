Pre-trained models
==================

whisper
-------

Currently, we support whisper multilingual models for spoken language identification.

.. list-table::

 * - Model type
   - Huggingface repo
 * - ``tiny``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny>`_
 * - ``base``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base>`_
 * - ``small``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small>`_
 * - ``medium``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium>`_

.. hint::

    You can also download them from

      `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_


In the following, we use the ``tiny`` model as an example. You can
replace ``tiny`` with ``base``, ``small``, or ``medium`` and everything still holds.

Download the model
^^^^^^^^^^^^^^^^^^

Please use the following commands to download the ``tiny`` model::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2

  # For Chinese users, please use
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2

  tar xvf sherpa-onnx-whisper-tiny.tar.bz2
  rm sherpa-onnx-whisper-tiny.tar.bz2

You should find the following files after unzipping::

  -rw-r--r--  1 fangjun  staff   427B Jan 31 16:21 README.md
  -rwxr-xr-x  1 fangjun  staff    19K Jan 31 16:21 export-onnx.py
  -rw-r--r--  1 fangjun  staff    15B Jan 31 16:21 requirements.txt
  -rwxr-xr-x  1 fangjun  staff    12K Jan 31 16:21 test.py
  drwxr-xr-x  6 fangjun  staff   192B Jan 31 16:22 test_wavs
  -rw-r--r--  1 fangjun  staff    86M Jan 31 16:22 tiny-decoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   109M Jan 31 16:22 tiny-decoder.onnx
  -rw-r--r--  1 fangjun  staff    12M Jan 31 16:22 tiny-encoder.int8.onnx
  -rw-r--r--  1 fangjun  staff    36M Jan 31 16:22 tiny-encoder.onnx
  -rw-r--r--  1 fangjun  staff   798K Jan 31 16:22 tiny-tokens.txt

Download test waves
^^^^^^^^^^^^^^^^^^^

Please use the following command to download test data::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2

  # For Chinese users, please use the following mirror
  # wget https://hub.nuaa.cf/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2

  tar xvf spoken-language-identification-test-wavs.tar.bz2
  rm spoken-language-identification-test-wavs.tar.bz2

You can find the following test files after unzipping::

  -rw-r--r--  1 fangjun  staff   222K Mar 24 12:51 ar-arabic.wav
  -rw-r--r--@ 1 fangjun  staff   137K Mar 24 13:09 bg-bulgarian.wav
  -rw-r--r--  1 fangjun  staff    83K Mar 24 13:07 cs-czech.wav
  -rw-r--r--  1 fangjun  staff   112K Mar 24 13:07 da-danish.wav
  -rw-r--r--  1 fangjun  staff   199K Mar 24 12:50 de-german.wav
  -rw-r--r--  1 fangjun  staff   207K Mar 24 13:06 el-greek.wav
  -rw-r--r--  1 fangjun  staff    31K Mar 24 12:45 en-english.wav
  -rw-r--r--@ 1 fangjun  staff    77K Mar 24 12:23 es-spanish.wav
  -rw-r--r--@ 1 fangjun  staff   371K Mar 24 12:21 fa-persian.wav
  -rw-r--r--  1 fangjun  staff   136K Mar 24 13:08 fi-finnish.wav
  -rw-r--r--  1 fangjun  staff   112K Mar 24 12:49 fr-french.wav
  -rw-r--r--  1 fangjun  staff   179K Mar 24 12:47 hi-hindi.wav
  -rw-r--r--@ 1 fangjun  staff   177K Mar 24 12:29 hr-croatian.wav
  -rw-r--r--  1 fangjun  staff   167K Mar 24 12:53 id-indonesian.wav
  -rw-r--r--  1 fangjun  staff   136K Mar 24 12:54 it-italian.wav
  -rw-r--r--  1 fangjun  staff    46K Mar 24 12:44 ja-japanese.wav
  -rw-r--r--@ 1 fangjun  staff   122K Mar 24 12:52 ko-korean.wav
  -rw-r--r--  1 fangjun  staff    85K Mar 24 12:54 nl-dutch.wav
  -rw-r--r--@ 1 fangjun  staff   241K Mar 24 12:38 no-norwegian.wav
  -rw-r--r--@ 1 fangjun  staff   121K Mar 24 12:35 po-polish.wav
  -rw-r--r--  1 fangjun  staff   166K Mar 24 12:48 pt-portuguese.wav
  -rw-r--r--@ 1 fangjun  staff   144K Mar 24 12:33 ro-romanian.wav
  -rw-r--r--  1 fangjun  staff   111K Mar 24 12:51 ru-russian.wav
  -rw-r--r--@ 1 fangjun  staff   239K Mar 24 12:40 sk-slovak.wav
  -rw-r--r--  1 fangjun  staff   196K Mar 24 13:01 sv-swedish.wav
  -rw-r--r--  1 fangjun  staff   106K Mar 24 13:14 ta-tamil.wav
  -rw-r--r--  1 fangjun  staff   104K Mar 24 13:02 tl-tagalog.wav
  -rw-r--r--  1 fangjun  staff    76K Mar 24 13:00 tr-turkish.wav
  -rw-r--r--  1 fangjun  staff   188K Mar 24 13:05 uk-ukrainian.wav
  -rw-r--r--  1 fangjun  staff   181K Mar 24 13:20 zh-chinese.wav

Test with Python APIs
^^^^^^^^^^^^^^^^^^^^^

After installing `sherpa-onnx`_ either from source or from using ``pip install sherpa-onnx``, you can run::

   python3 ./python-api-examples/spoken-language-identification.py \
     --whisper-encoder ./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx \
     --whisper-decoder ./sherpa-onnx-whisper-tiny/tiny-decoder.onnx \
     ./spoken-language-identification-test-wavs/de-german.wav

You should see the following output::


  2024-04-17 15:53:23,104 INFO [spoken-language-identification.py:158] File: ./spoken-language-identification-test-wavs/de-german.wav
  2024-04-17 15:53:23,104 INFO [spoken-language-identification.py:159] Detected language: de
  2024-04-17 15:53:23,104 INFO [spoken-language-identification.py:160] Elapsed seconds: 0.275
  2024-04-17 15:53:23,105 INFO [spoken-language-identification.py:161] Audio duration in seconds: 6.374
  2024-04-17 15:53:23,105 INFO [spoken-language-identification.py:162] RTF: 0.275/6.374 = 0.043


.. hint::

   You can find ``spoken-language-identification.py`` at

    `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/spoken-language-identification.py>`_

Android APKs
^^^^^^^^^^^^

You can find pre-built Android APKs for spoken language identification at the following address:

  `<https://k2-fsa.github.io/sherpa/onnx/spoken-language-identification/apk.html>`_

Huggingface space
^^^^^^^^^^^^^^^^^

We provide a huggingface space for spoken language identification.

You can visit the following URL:

  `<http://huggingface.co/spaces/k2-fsa/spoken-language-identification>`_

.. note::

  For Chinese users, you can use the following mirror:

    `<http://hf-mirror.com/spaces/k2-fsa/spoken-language-identification>`_
