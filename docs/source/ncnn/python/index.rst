.. _sherpa-ncnn-python-api:

Python API
==========

.. hint::

  It is known to work for ``Python >= 3.6`` on Linux, macOS, and Windows.

In this section, we describe

  1. How to install the Python package `sherpa-ncnn`_
  2. How to use `sherpa-ncnn`_ Python API for real-time speech recognition with
     a microphone
  3. How to use `sherpa-ncnn`_ Python API to recognize a single file

Installation
------------

You can use ``1`` of the  ``4`` methods below to install the Python package `sherpa-ncnn`_:

Method 1
^^^^^^^^

.. hint::

   This method works only on ``x86/x86_64`` systems: Linux, macOS and Windows.

   For other architectures, e.g., Mac M1, Raspberry Pi, etc, please
   use Method 2, 3, or 4.

.. code-block:: bash

  pip install sherpa-ncnn


If you use ``Method 1``, it will install pre-compiled libraries.
The ``disadvantage`` is that it may ``not be optimized`` for your platform,
while the ``advantage`` is that you don't need to install ``cmake`` or a
C++ compiler.

For the following methods, you have to first install:

- ``cmake``, which can be installed using ``pip install cmake``
- A C++ compiler, e.g., GCC on Linux and macOS, Visual Studio on Windows

Method 2
^^^^^^^^

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-ncnn
  cd sherpa-ncnn
  python3 setup.py install

Method 3
^^^^^^^^

.. code-block:: bash

  pip install git+https://github.com/k2-fsa/sherpa-ncnn


Method 4 (For developers and embedded boards)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. tabs::

   .. tab:: x86/x86_64

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-ncnn
        cd sherpa-ncnn
        mkdir build
        cd build

        cmake \
          -D SHERPA_NCNN_ENABLE_PYTHON=ON \
          -D SHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
          -D BUILD_SHARED_LIBS=ON \
          ..

        make -j6

        export PYTHONPATH=$PWD/lib:$PWD/../sherpa-ncnn/python:$PYTHONPATH

   .. tab:: 32-bit ARM

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-ncnn
        cd sherpa-ncnn
        mkdir build
        cd build

        cmake \
          -D SHERPA_NCNN_ENABLE_PYTHON=ON \
          -D SHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
          -D BUILD_SHARED_LIBS=ON \
          -DCMAKE_C_FLAGS="-march=armv7-a -mfloat-abi=hard -mfpu=neon" \
          -DCMAKE_CXX_FLAGS="-march=armv7-a -mfloat-abi=hard -mfpu=neon" \
          ..

        make -j6

        export PYTHONPATH=$PWD/lib:$PWD/../sherpa-ncnn/python:$PYTHONPATH

   .. tab:: 64-bit ARM

      .. code-block:: bash

        git clone https://github.com/k2-fsa/sherpa-ncnn
        cd sherpa-ncnn
        mkdir build
        cd build

        cmake \
          -D SHERPA_NCNN_ENABLE_PYTHON=ON \
          -D SHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
          -D BUILD_SHARED_LIBS=ON \
          -DCMAKE_C_FLAGS="-march=armv8-a" \
          -DCMAKE_CXX_FLAGS="-march=armv8-a" \
          ..

        make -j6

        export PYTHONPATH=$PWD/lib:$PWD/../sherpa-ncnn/python:$PYTHONPATH

Let us check whether `sherpa-ncnn`_ was installed successfully:

.. code-block:: bash

  python3 -c "import sherpa_ncnn; print(sherpa_ncnn.__file__)"
  python3 -c "import _sherpa_ncnn; print(_sherpa_ncnn.__file__)"

They should print the location of ``sherpa_ncnn`` and ``_sherpa_ncnn``.

.. hint::

  If you use ``Method 1``, ``Method 2``, and ``Method 3``, you can also use

    .. code-block:: bash

      python3 -c "import sherpa_ncnn; print(sherpa_ncnn.__version__)"

  It should print the version of `sherpa-ncnn`_, e.g., ``1.1``.


Next, we describe how to use `sherpa-ncnn`_ Python API for speech recognition:

  - (1) Real-time speech recognition with a microphone
  - (2) Recognize a file

Real-time recognition with a microphone
---------------------------------------

The following Python code shows how to use `sherpa-ncnn`_ Python API for
real-time speech recognition with a microphone.

.. hint::

  We use `sounddevice <https://python-sounddevice.readthedocs.io/en/0.4.5/>`_
  for recording. Please run ``pip install sounddevice`` before you run the
  code below.

.. note::

  You can download the code from

    `<https://github.com/k2-fsa/sherpa-ncnn/blob/master/python-api-examples/speech-recognition-from-microphone.py>`_

.. literalinclude:: ./code/speech-recognition-from-microphone.py
   :language: python
   :lines: 9-67
   :caption: Real-time speech recognition with a microphone using `sherpa-ncnn`_ Python API

**Code explanation**:

1. Import the required packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ./code/speech-recognition-from-microphone.py
   :language: python
   :lines: 11-21

Two packages are imported:

  - `sounddevice <https://python-sounddevice.readthedocs.io/en/0.4.5/>`_, for recording with a microphone
  - `sherpa-ncnn`_, for real-time speech recognition

2. Create the recognizer
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ./code/speech-recognition-from-microphone.py
   :language: python
   :lines: 24-38,40-43

We use the model :ref:`sherpa-ncnn-mixed-english-chinese-conv-emformer-model`
as an example, which is able to recognize both English and Chinese.
You can replace it with other pre-trained models.

Please refer to :ref:`sherpa-ncnn-pre-trained-models` for more models.

.. hint::

  The above example uses a ``float16`` encoder and joiner. You can also use
  the following code to switch to ``8-bit`` (i.e., ``int8``) quantized encoder
  and joiner.

    .. code-block:: python

      recognizer = sherpa_ncnn.Recognizer(
          tokens="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt",
          encoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.int8.param",
          encoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.int8.bin",
          decoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param",
          decoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin",
          joiner_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.int8.param",
          joiner_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.int8.bin",
          num_threads=4,
      )

3. Start recording
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ./code/speech-recognition-from-microphone.py
   :language: python
   :lines: 44,47

**Note that**:

  - We set channel to 1 since the model supports only a single channel
  - We use dtype ``float32`` so that the resulting audio samples are normalized
    to the range ``[-1, 1]``.
  - The sampling rate has to be ``recognizer.sample_rate``, which is 16 kHz for
    all models at present.

4. Read audio samples from the microphone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ./code/speech-recognition-from-microphone.py
   :language: python
   :lines: 45,49-50

**Note that**:

  - It reads ``100 ms`` of audio samples at a time. You can choose a larger
    value, e.g., ``200 ms``.
  - No queue or callback is used. Instead, we use a blocking read here.
  - The ``samples`` array is reshaped to a ``1-D`` array

5. Invoke the recognizer with audio samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ./code/speech-recognition-from-microphone.py
   :language: python
   :lines: 51

**Note that**:

  - ``samples`` has to be a 1-D tensor and should be normalized to the range
    ``[-1, 1]``.
  - Upon accepting the audio samples, the recognizer starts the decoding
    automatically. There is no separate call for decoding.

6. Get the recognition result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ./code/speech-recognition-from-microphone.py
   :language: python
   :lines: 52-55

We use ``recognizer.text`` to get the recognition result. To avoid
unnecessary output, we compare whether there is new result in ``recognizer.text``
and don't print to the console if there is nothing new recognized.

That's it!

Summary
^^^^^^^

In summary, you need to:

  1. Create the recognizer
  2. Start recording
  3. Read audio samples
  4. Call ``recognizer.accept_waveform(sample_rate, samples)``
  5. Call ``recognizer.text`` to get the recognition result

The following is a YouTube video for demonstration.

..  youtube:: 74SxVueROok
   :width: 120%


.. hint::

  If you don't have access to YouTube, please see the following video from bilibili:

  .. raw:: html

    <iframe src="//player.bilibili.com/player.html?bvid=BV1K44y197Fg&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" width="800" height="600"> </iframe>


.. note::

  `<https://github.com/k2-fsa/sherpa-ncnn/blob/master/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py>`_ supports endpoint detection.

  Please see the following video for its usage:

  .. raw:: html

    <iframe src="//player.bilibili.com/player.html?bvid=BV1eK411y788&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" width="800" height="600"> </iframe>


Recognize a file
----------------

The following Python code shows how to use `sherpa-ncnn`_ Python API to
recognize a wave file.

.. caution::

  The sampling rate of the wave file has to be 16 kHz. Also, it should
  contain only a single channel and samples should be 16-bit (i.e., int16)
  encoded.

.. note::

  You can download the code from

    `<https://github.com/k2-fsa/sherpa-ncnn/blob/master/python-api-examples/decode-file.py>`_

.. literalinclude:: ./code/decode-file.py
   :language: python
   :lines: 13-57
   :caption: Decode a file with `sherpa-ncnn`_ Python API

We use the model :ref:`sherpa-ncnn-mixed-english-chinese-conv-emformer-model`
as an example, which is able to recognize both English and Chinese.
You can replace it with other pre-trained models.

Please refer to :ref:`sherpa-ncnn-pre-trained-models` for more models.

.. hint::

  The above example uses a ``float16`` encoder and joiner. You can also use
  the following code to switch to ``8-bit`` (i.e., ``int8``) quantized encoder
  and joiner.

    .. code-block:: python

      recognizer = sherpa_ncnn.Recognizer(
          tokens="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt",
          encoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.int8.param",
          encoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.int8.bin",
          decoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param",
          decoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin",
          joiner_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.int8.param",
          joiner_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.int8.bin",
          num_threads=4,
      )
