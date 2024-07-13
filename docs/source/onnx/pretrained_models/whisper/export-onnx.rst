Export Whisper to ONNX
======================

This section describes how to export `Whisper`_ models to `onnx`_.


Available models
----------------

Note that we have already exported `Whisper`_ models to `onnx`_ and they are available
from the following huggingface repositories:

.. list-table::

 * - Model type
   - Huggingface repo
   - 国内用户
 * - ``tiny.en``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-tiny.en>`_
 * - ``base.en``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base.en>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-base.en>`_
 * - ``small.en``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small.en>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-small.en>`_
 * - ``distil-small.en``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-distil-small.en>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-distil-small.en>`_
 * - ``medium.en``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium.en>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-medium.en>`_
 * - ``distil-medium.en``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-distil-medium.en>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-distil-medium.en>`_
 * - ``tiny``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-tiny>`_
 * - ``base``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-base>`_
 * - ``small``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-small>`_
 * - ``medium``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-medium>`_
 * - ``large``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-large>`_
 * - ``large-v1``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large-v1>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-large-v1>`_
 * - ``large-v2``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large-v2>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-large-v2>`_
 * - ``large-v3``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large-v3>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-large-v3>`_
 * - ``distil-large-v2``
   - `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-distil-large-v2>`_
   - `点此 <https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-distil-large-v2>`_

.. note::

    You can also download them from

      `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_

    Models end with ``.en`` support only English and all
    other models are multilingual.


If you want to export the models by yourself or/and want to learn how the models
are exported, please read below.

Export to onnx
--------------

We use

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/whisper/export-onnx.py>`_

to export `Whisper`_ models to `onnx`_.

First, let us install dependencies and download the export script

.. code-block:: bash

   pip install torch openai-whisper onnxruntime onnx librosa soundfile

   git clone https://github.com/k2-fsa/sherpa-onnx/
   cd sherpa-onnx/scripts/whisper
   python3 ./export-onnx.py --help

It will print the following message:

.. code-block:: bash

  usage: export-onnx.py [-h] --model
                        {tiny,tiny.en,base,base.en,small,small.en,medium,medium.en,large,large-v1,large-v2,large-v3,distil-medium.en,distil-small.en,distil-large-v2,medium-aishell}

  optional arguments:
    -h, --help            show this help message and exit
    --model {tiny,tiny.en,base,base.en,small,small.en,medium,medium.en,large,large-v1,large-v2,large-v3,distil-medium.en,distil-small.en,distil-large-v2,medium-aishell}


Example 1: Export tiny.en
^^^^^^^^^^^^^^^^^^^^^^^^^

To export ``tiny.en``, we can use:

.. code-block:: bash

  python3 ./export-onnx.py --model tiny.en

It will generate the following files:

.. code-block:: bash

  (py38) fangjuns-MacBook-Pro:whisper fangjun$ ls -lh tiny.en-*
  -rw-r--r--  1 fangjun  staff   105M Aug  7 15:43 tiny.en-decoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   185M Aug  7 15:43 tiny.en-decoder.onnx
  -rw-r--r--  1 fangjun  staff    12M Aug  7 15:43 tiny.en-encoder.int8.onnx
  -rw-r--r--  1 fangjun  staff    36M Aug  7 15:43 tiny.en-encoder.onnx
  -rw-r--r--  1 fangjun  staff   816K Aug  7 15:43 tiny.en-tokens.txt

``tiny.en-encoder.onnx`` is the encoder model and ``tiny.en-decoder.onnx`` is the
decoder model.

``tiny.en-encoder.int8.onnx`` is the quantized encoder model and ``tiny.en-decoder.onnx`` is the
quantized decoder model.

``tiny.en-tokens.txt`` contains the token table, which maps an integer to a token and vice versa.

To check whether the exported model works correctly, we can use
  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/whisper/test.py>`_

We use `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/test_wavs/0.wav>`_
as the test wave.

.. code-block:: bash

   pip install kaldi-native-fbank
   wget https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/test_wavs/0.wav

   python3 ./test.py \
     --encoder ./tiny.en-encoder.onnx \
     --decoder ./tiny.en-decoder.onnx \
     --tokens ./tiny.en-tokens.txt \
     ./0.wav


To test ``int8`` quantized models, we can use:

.. code-block:: bash

   python3 ./test.py \
     --encoder ./tiny.en-encoder.int8.onnx \
     --decoder ./tiny.en-decoder.int8.onnx \
     --tokens ./tiny.en-tokens.txt \
     ./0.wav

Example 2: Export large-v3
^^^^^^^^^^^^^^^^^^^^^^^^^^

To export ``large-v3``, we can use:

.. code-block:: bash

  python3 ./export-onnx.py --model large-v3

It will generate the following files:

.. code-block:: bash

  (py38) fangjuns-MacBook-Pro:whisper fangjun$ ls -lh large-v3-*
  -rw-r--r--  1 fangjun  staff   2.7M Jul 12 20:38 large-v3-decoder.onnx
  -rw-r--r--  1 fangjun  staff   3.0G Jul 12 20:38 large-v3-decoder.weights
  -rw-r--r--  1 fangjun  staff   744K Jul 12 20:35 large-v3-encoder.onnx
  -rw-r--r--  1 fangjun  staff   2.8G Jul 12 20:35 large-v3-encoder.weights
  -rw-r--r--  1 fangjun  staff   798K Jul 12 20:32 large-v3-tokens.txt

``large-v3-encoder.onnx`` is the encoder model and ``large-v3-decoder.onnx`` is the
decoder model.

Note that for ``large`` models, there will also be two additional ``weights`` files.

``large-v3-tokens.txt`` contains the token table, which maps an integer to a token and vice versa.

To check whether the exported model works correctly, we can use
  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/whisper/test.py>`_

We use `<https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/test_wavs/0.wav>`_
as the test wave.

.. code-block:: bash

   pip install kaldi-native-fbank
   wget https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/test_wavs/0.wav

   python3 ./test.py \
     --encoder ./large-v3-encoder.onnx \
     --decoder ./large-v3-decoder.onnx \
     --tokens ./large-v3-tokens.txt \
     ./0.wav

.. hint::

   We provide a colab notebook
   |sherpa-onnx with whisper large-v3 colab notebook|
   for you to try the exported large-v3 onnx model with sherpa-onnx
   on CPU as well as on GPU.

   You will find the RTF on GPU (Tesla T4) is less than 1.

.. |sherpa-onnx with whisper large-v3 colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_whisper_large_v3.ipynb
