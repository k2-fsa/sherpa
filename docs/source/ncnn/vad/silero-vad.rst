silero-vad
==========

We support only `silero-vad`_ v4 in `sherpa-ncnn`_.

.. _sherpa-ncnn-silero-vad:

sherpa-ncnn-silero-vad
----------------------

This model is converted from `silero-vad`_ v4 using scripts from

  `<https://github.com/wxqwinner/silero-vad-ncnn>`_

Download the model
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd /path/to/sherpa-ncnn

   wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-silero-vad.tar.bz2
   tar xvf sherpa-ncnn-silero-vad.tar.bz2
   rm sherpa-ncnn-silero-vad.tar.bz2

.. code-block:: bash

  ls -lh sherpa-ncnn-silero-vad/

  total 640
  -rw-r--r--  1 fangjun  staff   247B Aug 17  2024 README.md
  -rw-r--r--  1 fangjun  staff   305K Aug 17  2024 silero.ncnn.bin
  -rw-r--r--  1 fangjun  staff   5.2K Aug 17  2024 silero.ncnn.param

C++ examples
~~~~~~~~~~~~

.. list-table::

 * -
   - URL
 * - Remove silences from a file
   - `sherpa-ncnn-vad.cc <https://github.com/k2-fsa/sherpa-ncnn/blob/master/sherpa-ncnn/csrc/sherpa-ncnn-vad.cc>`_
 * - VAD + microphone to record speeches
   - `sherpa-ncnn-vad-microphone.cc <https://github.com/k2-fsa/sherpa-ncnn/blob/master/sherpa-ncnn/csrc/sherpa-ncnn-vad-microphone.cc>`_
 * - VAD + ASR
   - `sherpa-ncnn-vad-microphone-offline-asr.cc <https://github.com/k2-fsa/sherpa-ncnn/blob/master/sherpa-ncnn/csrc/sherpa-ncnn-vad-microphone-offline-asr.cc>`_

 * - VAD + real-time ASR
   - `sherpa-ncnn-vad-microphone-simulated-streaming-asr.cc <https://github.com/k2-fsa/sherpa-ncnn/blob/master/sherpa-ncnn/csrc/sherpa-ncnn-vad-microphone-simulated-streaming-asr.cc>`_

Please read the help info in the above example files for usages.
