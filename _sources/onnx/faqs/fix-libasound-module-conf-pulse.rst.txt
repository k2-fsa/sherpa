Cannot open shared library libasound_module_conf_pulse.so
=========================================================

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

