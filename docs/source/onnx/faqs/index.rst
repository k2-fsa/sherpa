Frequently Asked Question (FAQs)
================================

This page contains frequently asked questions for `sherpa-onnx`_.

在线、离线、流式、非流式的区别
------------------------------

此项目中，``在线`` 等同于流式，``离线`` 等同于非流式。

``在线`` 即流式，是边说边识别；响应速度快、延迟小。

``离线`` 即非流式，是把所有待识别的数据，一次性送给模型；特点是需要
等待所有的数据都到齐, 然后才能开始识别。

不管是 ``离线`` 还是 ``在线``, 我们这个项目，都不需要访问网络，都可以在本地
处理；即使断网，也能正常工作。

Cannot open shared library libasound_module_conf_pulse.so
---------------------------------------------------------

The detailed errors are given below:

.. code-block::

  Cannot open shared library libasound_module_conf_pulse.so
  (/usr/lib64/alsa-lib/libasound_module_conf_pulse.so: cannot open shared object file: No such file or directory)
  ALSA lib pcm.c:2722:(snd_pcm_open_noupdate) Unknown PCM

If you use Linux and get the above error when trying to use the microphone, please do the following:

  1. Locate where is the file ``libasound_module_conf_pulse.so`` on your system

    .. code-block:: bash

      find / -name libasound_module_conf_pulse.so 2>/dev/null

  2. If the above search command prints::

      /usr/lib/x86_64-linux-gnu/alsa-lib/libasound_module_conf_pulse.so
      /usr/lib/i386-linux-gnu/alsa-lib/libasound_module_conf_pulse.so

  3. Please run::

      sudo mkdir -p /usr/lib64/alsa-lib
      sudo ln -s /usr/lib/x86_64-linux-gnu/alsa-lib/libasound_module_conf_pulse.so /usr/lib64/alsa-lib

  4. Now your issue should be fixed.


TTS 中文模型没有声音
--------------------

Please see :ref:`how_to_enable_utf8_on_windows`.
You need to use ``UTF-8`` encoding for your system.

./gitcompile: line 89: libtoolize: command not found
----------------------------------------------------

If you are using Linux and get the following error:

.. code-block::

   ./gitcompile: line 89: libtoolize: command not found

Please run::

  sudo apt-get install libtool

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
