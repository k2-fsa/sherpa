Speech recognition from URLs
============================

`sherpa-onnx`_ also supports decoding from URLs.

.. hint::

   Only streaming models are currently supported. Please modify the
   `code <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-url.py>`_
   for non-streaming models on need.

All types of URLs supported by ``ffmpeg`` are supported.

The following table lists some example URLs.

.. list-table::

  * - Type
    - Example
  * - `RTMP`_
    - ``rtmp://localhost/live/livestream``
  * - OPUS file
    - `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/wenetspeech/DEV_T0000000000.opus>`_
  * - WAVE file
    - `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/aishell2/ID0012W0030.wav>`_
  * - Local WAVE file
    - ``file:///Users/fangjun/open-source/sherpa-onnx/a.wav``

Before you continue, please install ``ffmpeg`` first.

For instance, you can use ``sudo apt-get install ffmpeg`` for Ubuntu
and ``brew install ffmpeg`` for macOS.

We use the model :ref:`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`
for demonstration in the following examples.

Decode a URL
------------

This example shows you how to decode a URL pointing to a file.

.. hint::

   The file does not need to be a WAVE file. It can be a file of any format
   supported by ``ffmpeg``.

.. code-block:: bash

  python3 ./python-api-examples/speech-recognition-from-url.py \
    --encoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --decoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
    --tokens ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    --url https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition/resolve/main/test_wavs/librispeech/1089-134686-0001.wav

RTMP
----

In this example, we use ``ffmpeg`` to capture a microphone and push the
audio stream to a server using RTMP, and then we start `sherpa-onnx`_ to pull
the audio stream from the server for recognition.

Install the server
~~~~~~~~~~~~~~~~~~

We will use `srs`_ as the server. Let us first install `srs`_ from source:

.. code-block:: bash

  git clone -b develop https://github.com/ossrs/srs.git
  cd srs/trunk
  ./configure
  make

  # Check that we have compiled srs successfully
  ./objs/srs --help

  # Note: ./objs/srs is statically linked and depends only on system libraries.

Start the server
~~~~~~~~~~~~~~~~

.. code-block:: bash

   ulimit -HSn 10000

   # switch to the directory srs/trunk
   ./objs/srs -c conf/srs.conf

The above command gives the following output:

.. code-block:: bash

  srs(51047,0x7ff8451198c0) malloc: nano zone abandoned due to inability to preallocate reserved vm space.
  Asan: Please setup the env MallocNanoZone=0 to disable the warning, see https://stackoverflow.com/a/70209891/17679565
  [2023-07-05 12:19:23.017][INFO][51047][78gw8v44] XCORE-SRS/6.0.55(Bee)
  [2023-07-05 12:19:23.021][INFO][51047][78gw8v44] config parse complete
  [2023-07-05 12:19:23.021][INFO][51047][78gw8v44] you can check log by: tail -n 30 -f ./objs/srs.log
  [2023-07-05 12:19:23.021][INFO][51047][78gw8v44] please check SRS by: ./etc/init.d/srs status


To check the status of `srs`_, use

.. code-block:: bash

   ./etc/init.d/srs status

which gives the following output:

.. code-block:: bash

   SRS(pid 51548) is running.                                 [  OK  ]

.. hint::

   If you fail to start the `srs`_ server, please check the log file
   ``./objs/srs.log`` for a fix.

Start ffmpeg to push audio stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let us list available recording devices on the current computer
with the following command:

.. code-block:: bash

  ffmpeg -hide_banner -f avfoundation -list_devices true -i ""

It gives the following output on my computer:

.. code-block:: bash

  [AVFoundation indev @ 0x7f9f41904840] AVFoundation video devices:
  [AVFoundation indev @ 0x7f9f41904840] [0] FaceTime HD Camera (Built-in)
  [AVFoundation indev @ 0x7f9f41904840] [1] Capture screen 0
  [AVFoundation indev @ 0x7f9f41904840] AVFoundation audio devices:
  [AVFoundation indev @ 0x7f9f41904840] [0] Background Music
  [AVFoundation indev @ 0x7f9f41904840] [1] MacBook Pro Microphone
  [AVFoundation indev @ 0x7f9f41904840] [2] Background Music (UI Sounds)
  [AVFoundation indev @ 0x7f9f41904840] [3] WeMeet Audio Device
  : Input/output error

We will use the device ``[1] MacBook Pro Microphone``. Note that its index
is ``1``, so we will use ``-i ":1"`` in the following command to start
recording and push the recorded audio stream to the server under the
address ``rtmp://localhost/live/livestream``.

.. hint::

   The default TCP port for `RTMP`_ is ``1935``.

.. code-block:: bash

  ffmpeg -hide_banner -f avfoundation -i ":1" -acodec aac -ab 64k -ar 16000 -ac 1 -f flv rtmp://localhost/live/livestream

The above command gives the following output:

.. code-block:: bash

    Input #0, avfoundation, from ':1':
      Duration: N/A, start: 830938.803938, bitrate: 1536 kb/s
      Stream #0:0: Audio: pcm_f32le, 48000 Hz, mono, flt, 1536 kb/s
    Stream mapping:
      Stream #0:0 -> #0:0 (pcm_f32le (native) -> aac (native))
    Press [q] to stop, [?] for help
    Output #0, flv, to 'rtmp://localhost/live/livestream':
      Metadata:
        encoder         : Lavf60.3.100
      Stream #0:0: Audio: aac (LC) ([10][0][0][0] / 0x000A), 16000 Hz, mono, fltp, 64 kb/s
        Metadata:
          encoder         : Lavc60.3.100 aac
    size=      64kB time=00:00:08.39 bitrate=  62.3kbits/s speed=0.977x


Start sherpa-onnx to pull audio stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can start `sherpa-onnx`_ to pull audio stream from ``rtmp://localhost/live/livestream``
for speech recognition.

.. code-block:: bash

  python3 ./python-api-examples/speech-recognition-from-url.py \
    --encoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --decoder ./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
    --joiner ./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
    --tokens ./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
    --url rtmp://localhost/live/livestream

You should see the recognition result printed to the console as you speak.

.. hint::

   You can replace ``localhost`` with your server IP
   and start `sherpa-onnx`_ on many computers at the same time to pull
   audio stream from the address `<rtmp://your_server_ip/live/livestream>`_.
