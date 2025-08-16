.. _sherpa_onnx_offline_nemo_transducer_models:

NeMo transducer-based Models
============================

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.


.. _sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8:

sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 (25 European Languages)
----------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt-0.6b-v3>`_

It supports 25 European languages:

  - Bulgarian (bg), Croatian (hr), Czech (cs), Danish (da), Dutch (nl)
  - English (en), Estonian (et), Finnish (fi), French (fr), German (de)
  - Greek (el), Hungarian (hu), Italian (it), Latvian (lv), Lithuanian (lt)
  - Maltese (mt), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk)
  - Slovenian (sl), Spanish (es), Swedish (sv), Russian (ru), Ukrainian (uk)

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Colab
~~~~~

We provide two colab notebooks for this model:

  - `Colab with CPU <https://colab.research.google.com/drive/1ixBBirCv7vOcM0QNITwad9iSGFpG5an4?usp=sharing>`_
  - `Colab with NVIDIA GPU <https://colab.research.google.com/drive/1EUgBbM165YZLnef2iYf_ZIv6mBn9GhLG?usp=sharing>`_

Huggingface space
~~~~~~~~~~~~~~~~~

You can try it by visiting `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
   tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
   rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/

  total 640M
  -rw-r--r-- 1 501 staff  12M Aug 16 09:00 decoder.int8.onnx
  -rw-r--r-- 1 501 staff 622M Aug 16 09:00 encoder.int8.onnx
  -rw-r--r-- 1 501 staff 6.1M Aug 16 09:00 joiner.int8.onnx
  drwxr-xr-x 2 501 staff 4.0K Aug 16 09:00 test_wavs
  -rw-r--r-- 1 501 staff  92K Aug 16 09:00 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  build/bin/sherpa-onnx-offline \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/test_wavs/en.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-nemo/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
    --model-type=nemo_transducer

.. _sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8:

sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 (English, 英语)
----------------------------------------------------------------------

This model is converted from

  `<https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/parakeet-tdt-0.6b-v2>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

.. hint::

   This model supports punctuations and cases.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
   tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
   rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2

.. hint::

   If you want to try ``float16`` quantized model, please use  `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2>`_.

   If you want to try ``non-quantized`` decoder and joiner models, please use `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2.tar.bz2>`_

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/
  total 1295752
  -rw-r--r--  1 fangjun  staff   6.9M May  6 16:24 decoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   622M May  6 16:24 encoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   1.7M May  6 16:24 joiner.int8.onnx
  drwxr-xr-x  3 fangjun  staff    96B May  6 16:24 test_wavs
  -rw-r--r--  1 fangjun  staff   9.2K May  6 16:24 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

You should see the following output:

.. literalinclude:: ./code-nemo/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt \
    --model-type=nemo_transducer

RTF on RK3588 with Cortex A76 CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following, we test this model on RK3588 with Cortex A76 CPU.

Information about the CPUs on the board is given below:

.. literalinclude:: ./code-nemo/rk3588-cpu.txt

You can see that it has 8 CPUs: 4 Cortex A55 + 4 Cortex A76.

We use ``taskset`` below to test the RTF on Cortex A76.

.. code-block:: bash

  taskset 0x80 sherpa-onnx-offline \
    --num-threads=1 \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav

Its output is given below:

.. literalinclude:: ./code-nemo/rk3588-a76-rtf.txt


To test the RTF with different ``--num-threads``, we use::

  taskset 0xc0 sherpa-onnx-offline \
    --num-threads=2 \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav

  taskset 0xe0 sherpa-onnx-offline \
    --num-threads=3 \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav

  taskset 0xf0 sherpa-onnx-offline \
    --num-threads=4 \
    --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx \
    --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx \
    --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav

The results are summarized below:

.. list-table::

 * - Number of threads
   - 1
   - 2
   - 3
   - 4
 * - RTF on Cortex A76 CPU
   - 0.220
   - 0.142
   - 0.118
   - 0.088


sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19 (Russian, 俄语)
--------------------------------------------------------------------------------

This model is converted from

  `<https://github.com/salute-developers/GigaAM>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/GigaAM/run-rnnt-v2.sh>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.


Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
  tar xvf sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
  rm sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19

  total 231M
  -rw-r--r-- 1 501 staff 3.2M Apr 20 01:58 decoder.onnx
  -rw-r--r-- 1 501 staff 226M Apr 20 01:59 encoder.int8.onnx
  -rw-r--r-- 1 501 staff 1.4M Apr 20 01:58 joiner.onnx
  -rw-r--r-- 1 501 staff 219K Apr 20 01:59 LICENSE
  -rw-r--r-- 1 501 staff  302 Apr 20 01:59 README.md
  -rwxr-xr-x 1 501 staff  868 Apr 20 01:51 run-rnnt-v2.sh
  -rwxr-xr-x 1 501 staff 8.9K Apr 20 01:59 test-onnx-rnnt.py
  drwxr-xr-x 2 501 staff 4.0K Apr 21 09:35 test_wavs
  -rw-r--r-- 1 501 staff  196 Apr 20 01:58 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/test_wavs/example.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-nemo/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/tokens.txt \
    --model-type=nemo_transducer


sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24 (Russian, 俄语)
--------------------------------------------------------------------------------

This model is converted from

  `<https://github.com/salute-developers/GigaAM>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/GigaAM/run-rnnt.sh>`_

.. warning::

   The license of the model can be found at `<https://github.com/salute-developers/GigaAM/blob/main/GigaAM%20License_NC.pdf>`_.

   It is for non-commercial use only.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.tar.bz2
  tar xvf sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.tar.bz2
  rm sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/
  total 548472
  -rw-r--r--  1 fangjun  staff    89K Oct 25 13:36 GigaAM%20License_NC.pdf
  -rw-r--r--  1 fangjun  staff   318B Oct 25 13:37 README.md
  -rw-r--r--  1 fangjun  staff   3.8M Oct 25 13:36 decoder.onnx
  -rw-r--r--  1 fangjun  staff   262M Oct 25 13:37 encoder.int8.onnx
  -rw-r--r--  1 fangjun  staff   3.8K Oct 25 13:32 export-onnx-rnnt.py
  -rw-r--r--  1 fangjun  staff   2.0M Oct 25 13:36 joiner.onnx
  -rwxr-xr-x  1 fangjun  staff   2.0K Oct 25 13:32 run-rnnt.sh
  -rwxr-xr-x  1 fangjun  staff   8.7K Oct 25 13:32 test-onnx-rnnt.py
  drwxr-xr-x  4 fangjun  staff   128B Oct 25 13:37 test_wavs
  -rw-r--r--  1 fangjun  staff   5.8K Oct 25 13:36 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/tokens.txt \
    --model-type=nemo_transducer \
    ./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/test_wavs/example.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-nemo/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/tokens.txt \
    --model-type=nemo_transducer

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --encoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/encoder.int8.onnx \
    --decoder=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/decoder.onnx \
    --joiner=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/joiner.onnx \
    --tokens=./sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24/tokens.txt \
    --model-type=nemo_transducer
