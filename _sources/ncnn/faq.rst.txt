FAQs
====

Where to get help
-----------------

If you have any questions, please create an issue
at `<https://github.com/k2-fsa/sherpa-ncnn>`_

We also have active social groups:

  - 微信公众号: 新一代 Kaldi
  - 微信交流群：请关注新一代 Kaldi, 添加工作人员微信, 我们邀请您进群
  - QQ 群：744602236


No default input device found
-----------------------------

If you are using Linux and if ``sherpa-ncnn-microphone`` throws the following error:

.. code-block::

   Num device: 0
   No default input device found.

Please consider using ``sherpa-ncnn-alsa`` to replace ``sherpa-ncnn-microphone``.
If you cannot find ``sherpa-ncnn-alsa`` in ``./build/bin``, please run the
following commands:

.. code-block:: bash

  cd /path/to/sherpa-ncnn
  sudo apt-get install alsa-utils libasound2-dev
  cd build
  rm CMakeCache.txt # Important, remove the cmake cache file
  make -j

After the above commands, you should see a binary file ``./build/bin/sherpa-ncnn-alsa``.


Please follow :ref:`sherpa-ncnn-alsa` to use ``sherpa-ncnn-alsa``.
