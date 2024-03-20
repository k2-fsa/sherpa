Frequently Asked Question (FAQs)
================================

This page contains frequently asked questions for `sherpa-onnx`_.

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

