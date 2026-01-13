Frequently Asked Question (FAQs)
================================

This page contains frequently asked questions for `sherpa-onnx`_.

.. toctree::
   :maxdepth: 5

   ./diff-online-offline.rst
   ./change-kotlin-and-java-package-name.rst
   ./fix-libasound-module-conf-pulse.rst
   ./fix-tts-encoding-for-chinese-models.rst
   ./fix-libtoolize.rst
   ./static-onnxruntime-linux-x64.rst


OSError: PortAudio library not found
------------------------------------

If you have the following error on Linux (Ubuntu),

.. code-block:: bash

  Traceback (most recent call last):
    File "/mnt/sdb/shared/sherpa-onnx/./python-api-examples/vad-microphone.py", line 8, in <module>
      import sounddevice as sd
    File "/mnt/sdb/shared/py311/lib/python3.11/site-packages/sounddevice.py", line 71, in <module>
      raise OSError('PortAudio library not found')
  OSError: PortAudio library not found

Then please run::

  sudo apt-get install libportaudio2

and then re-try.

imports github.com/k2-fsa/sherpa-onnx-go-linux: build constraints exclude all Go files
--------------------------------------------------------------------------------------

If you have the following output when running ``go build``::

  [root@VM-0-3-centos non-streaming-decode-files]# go build
  package non-streaming-decode-files
   imports github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx
   imports github.com/k2-fsa/sherpa-onnx-go-linux: build constraints exclude all Go files in /root/go/pkg/mod/github.com/k2-fsa/sherpa-onnx-go-linux@v1.9.21

Please first run::

  go env -w CGO_ENABLED=1

And then re-run ``go build``.

External buffers are not allowed
--------------------------------

If you are using ``electron >= 21`` and get the following error:

.. code-block::

   External buffers are not allowed

Then please set ``enableExternalBuffer`` to ``false``.

Specifically,

  - For reading wave files, please use ``sherpa_onnx.readWave(filename, false);``,
    where the second argument ``false`` means to not use external buffers

  - For VAD, please use ``vad.get(startIndex, n, false)`` and ``vad.front(false)``

  - For speaker identification, please use ``extractor.compute(stream, false)``

  - For TTS, please use:

    .. code-block:: javascript

        const audio = tts.generate({
          text: text,
          sid: 0,
          speed: 1.0,
          enableExternalBuffer: false,
        });

The given version [17] is not supported, only version 1 to 10 is supported in this build
----------------------------------------------------------------------------------------

If you have such an error, please find the file ``onnxruntime.dll`` in your ``C`` drive
and try to remove it.

The reason is that you have two ``onnxruntime.dll`` on your computer and the one
in your ``C`` drive is outdated.
