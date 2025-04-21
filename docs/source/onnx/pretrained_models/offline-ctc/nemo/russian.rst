Russian
=======

.. hint::

   Please refer to :ref:`install_sherpa_onnx` to install `sherpa-onnx`_
   before you read this section.

This page lists offline CTC models from `NeMo`_ for Russian.

sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19
--------------------------------------------------

This model is converted from

  `<https://github.com/salute-developers/GigaAM>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/GigaAM/run-ctc-v2.sh>`_

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.tar.bz2
   tar xvf sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.tar.bz2
   rm sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/

  total 226M
  -rwxr-xr-x 1 501 staff 1.8K Apr 20 01:51 export-onnx-ctc-v2.py
  -rw-r--r-- 1 501 staff 219K Apr 20 01:57 LICENSE
  -rw-r--r-- 1 501 staff 226M Apr 20 01:57 model.int8.onnx
  -rwxr-xr-x 1 501 staff  866 Apr 20 01:51 run-ctc-v2.sh
  -rwxr-xr-x 1 501 staff 4.1K Apr 20 01:57 test-onnx-ctc.py
  drwxr-xr-x 2 501 staff 4.0K Apr 21 09:43 test_wavs
  -rw-r--r-- 1 501 staff  196 Apr 20 01:57 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/tokens.txt \
    ./sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/test_wavs/example.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-russian/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/tokens.txt

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19/tokens.txt

sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24
-----------------------------------------------

This model is converted from

  `<https://github.com/salute-developers/GigaAM>`_

You can find the conversion script at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/GigaAM/run-ctc.sh>`_

.. warning::

   The license of the model can be found at `<https://github.com/salute-developers/GigaAM/blob/main/GigaAM%20License_NC.pdf>`_.

   It is for non-commercial use only.

In the following, we describe how to download it and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following commands to download it.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24.tar.bz2
   tar xvf sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24.tar.bz2
   rm sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24.tar.bz2

You should see something like below after downloading::

  ls -lh sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24/

  total 558904
  -rw-r--r--  1 fangjun  staff    89K Oct 24 21:20 GigaAM%20License_NC.pdf
  -rw-r--r--  1 fangjun  staff   318B Oct 24 21:20 README.md
  -rwxr-xr-x  1 fangjun  staff   3.5K Oct 24 21:20 export-onnx-ctc.py
  -rw-r--r--  1 fangjun  staff   262M Oct 24 21:24 model.int8.onnx
  -rwxr-xr-x  1 fangjun  staff   1.2K Oct 24 21:20 run-ctc.sh
  -rwxr-xr-x  1 fangjun  staff   4.1K Oct 24 21:20 test-onnx-ctc.py
  drwxr-xr-x  4 fangjun  staff   128B Oct 24 21:24 test_wavs
  -rw-r--r--@ 1 fangjun  staff   196B Oct 24 21:31 tokens.txt

Decode wave files
~~~~~~~~~~~~~~~~~

.. hint::

   It supports decoding only wave files of a single channel with 16-bit
   encoded samples, while the sampling rate does not need to be 16 kHz.

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-offline \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24/tokens.txt \
    ./sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24/test_wavs/example.wav

.. note::

   Please use ``./build/bin/Release/sherpa-onnx-offline.exe`` for Windows.

.. caution::

   If you use Windows and get encoding issues, please run:

      .. code-block:: bash

          CHCP 65001

   in your commandline.

You should see the following output:

.. literalinclude:: ./code-russian/sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24.int8.txt

Speech recognition from a microphone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  ./build/bin/sherpa-onnx-microphone-offline \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24/tokens.txt

Speech recognition from a microphone with VAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  ./build/bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=./silero_vad.onnx \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24/model.int8.onnx \
    --tokens=./sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24/tokens.txt
