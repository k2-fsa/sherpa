.. _sherpa-onnx-rknn-install:

Install
=======

You can use any methods below to build and install `sherpa-onnx`_ for RKNPU.

From pre-built wheels using pip install
---------------------------------------

You can find pre-built ``whl`` files at  `<https://k2-fsa.github.io/sherpa/onnx/rk-npu.html>`_.

To install it, you can use:

.. code-block:: bash

   pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu.html

   # For Chinese users
   pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html

To check that you have installed `sherpa-onnx`_ with rknn support, please run

.. code-block:: bash

  (py310) orangepi@orangepi5max:~/t$ ldd $(which sherpa-onnx)
    linux-vdso.so.1 (0x0000007f9fd93000)
    librknnrt.so => /lib/librknnrt.so (0x0000007f9f480000)
    libonnxruntime.so => /home/orangepi/py310/bin/../lib/python3.10/site-packages/sherpa_onnx/lib/libonnxruntime.so (0x0000007f9e7f0000)
    libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6 (0x0000007f9e750000)
    libstdc++.so.6 => /lib/aarch64-linux-gnu/libstdc++.so.6 (0x0000007f9e520000)
    libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1 (0x0000007f9e4f0000)
    libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (0x0000007f9e340000)
    /lib/ld-linux-aarch64.so.1 (0x0000007f9fd5a000)
    libpthread.so.0 => /lib/aarch64-linux-gnu/libpthread.so.0 (0x0000007f9e320000)
    libdl.so.2 => /lib/aarch64-linux-gnu/libdl.so.2 (0x0000007f9e300000)
    librt.so.1 => /lib/aarch64-linux-gnu/librt.so.1 (0x0000007f9e2e0000)

You should check that ``librknnrt.so`` is in the dependency list.

If you cannot find ``librknnrt.so``, it means you have failed to install `sherpa-onnx`_
with rknn support. In that case, please visit

  - `<https://k2-fsa.github.io/sherpa/onnx/rk-npu.html>`_
  - For Chinese users `<https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html>`_

and download the ``whl`` file to your board and use ``pip install ./*.whl``
to install from ``whl``. Remember to recheck with ``ldd $(which sherpa-onnx)``.

**Caution**: Please use the following command to check the version of your ``librknnrt.so``:

.. code-block:: bash

  strings /lib/librknnrt.so | grep "librknnrt version"

It should print something like below::

  librknnrt version: 2.2.0 (c195366594@2024-09-14T12:18:56)

``librknnrt.so`` ``2.2.0`` is known to work on rk3588.

You can download ``librknnrt.so`` ``2.2.0`` from:

  `<https://github.com/airockchip/rknn-toolkit2/blob/v2.2.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so>`_

.. caution::

   ``/lib/librknnrt.so`` is used in the above since it is in the output of ``ldd $(which sherpa-onnx)``.

   You need to change it to match your case. For instance, if it is ``/usr/lib/librknnrt.so``
   in your output, then you should use::

      strings /usr/lib/librknnrt.so | grep "librknnrt version"

Download pre-compiled binaries
----------------------------------------

You can download pre-compiled binaries from

  `<https://github.com/k2-fsa/sherpa-onnx/releases>`_

Please always use the ``latest version``. For the version ``v1.12.13``, you can visit

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.12.13>`_

and select

  - Static link: `sherpa-onnx-v1.12.13-rknn-linux-aarch64-static.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.13/sherpa-onnx-v1.12.13-rknn-linux-aarch64-static.tar.bz2>`_
  - Dynamic link: `sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.13/sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared.tar.bz2>`_

We use the dynamically linked binaries as an example below.

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.13/sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared.tar.bz2
   tar xvf sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared.tar.bz2

.. code-block:: bash

  orangepi@orangepi5max:~$ ls -lh sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared
  total 8.0K
  drwxr-xr-x 2 orangepi orangepi 4.0K Sep 12  2025 bin
  drwxr-xr-x 2 orangepi orangepi 4.0K Sep 12  2025 lib
  orangepi@orangepi5max:~$ ls -lh sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared/lib/
  total 18M
  -rw-r--r-- 1 orangepi orangepi  13M Sep 12  2025 libonnxruntime.so
  -rwxr-xr-x 1 orangepi orangepi 4.7M Sep 12  2025 libsherpa-onnx-c-api.so
  -rwxr-xr-x 1 orangepi orangepi 217K Sep 12  2025 libsherpa-onnx-cxx-api.so
  orangepi@orangepi5max:~$ ls  sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared/bin/
  sherpa-onnx                                            sherpa-onnx-offline                          sherpa-onnx-offline-zeroshot-tts
  sherpa-onnx-alsa                                       sherpa-onnx-offline-audio-tagging            sherpa-onnx-online-punctuation
  sherpa-onnx-alsa-offline                               sherpa-onnx-offline-denoiser                 sherpa-onnx-online-websocket-client
  sherpa-onnx-alsa-offline-audio-tagging                 sherpa-onnx-offline-language-identification  sherpa-onnx-online-websocket-server
  sherpa-onnx-alsa-offline-speaker-identification        sherpa-onnx-offline-parallel                 sherpa-onnx-vad
  sherpa-onnx-keyword-spotter                            sherpa-onnx-offline-punctuation              sherpa-onnx-vad-alsa
  sherpa-onnx-keyword-spotter-alsa                       sherpa-onnx-offline-source-separation        sherpa-onnx-vad-alsa-offline-asr
  sherpa-onnx-keyword-spotter-microphone                 sherpa-onnx-offline-speaker-diarization      sherpa-onnx-vad-microphone
  sherpa-onnx-microphone                                 sherpa-onnx-offline-tts                      sherpa-onnx-vad-microphone-offline-asr
  sherpa-onnx-microphone-offline                         sherpa-onnx-offline-tts-play                 sherpa-onnx-vad-with-offline-asr
  sherpa-onnx-microphone-offline-audio-tagging           sherpa-onnx-offline-tts-play-alsa            sherpa-onnx-vad-with-online-asr
  sherpa-onnx-microphone-offline-speaker-identification  sherpa-onnx-offline-websocket-server         sherpa-onnx-version

.. code-block:: bash

  orangepi@orangepi5max:~$ ldd sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared/bin/sherpa-onnx

          linux-vdso.so.1 (0x0000007fae61e000)
          librknnrt.so (0x0000007fadee0000)
          libonnxruntime.so (0x0000007fad250000)
          libpthread.so.0 => /lib/aarch64-linux-gnu/libpthread.so.0 (0x0000007fad230000)
          libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6 (0x0000007fad190000)
          libstdc++.so.6 => /lib/aarch64-linux-gnu/libstdc++.so.6 (0x0000007facf60000)
          libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1 (0x0000007facf30000)
          libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (0x0000007facd80000)
          libdl.so.2 => /lib/aarch64-linux-gnu/libdl.so.2 (0x0000007facd60000)
          librt.so.1 => /lib/aarch64-linux-gnu/librt.so.1 (0x0000007facd40000)
          /lib/ld-linux-aarch64.so.1 (0x0000007fae5e5000)


.. code-block:: bash

  orangepi@orangepi5max:~$ readelf -d sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared/bin/sherpa-onnx

  Dynamic section at offset 0x1dfc20 contains 31 entries:
    Tag        Type                         Name/Value
   0x0000000000000001 (NEEDED)             Shared library: [librknnrt.so]
   0x0000000000000001 (NEEDED)             Shared library: [libonnxruntime.so]
   0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
   0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
   0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
   0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
   0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
   0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../../sherpa_onnx/lib]
   0x000000000000000c (INIT)               0x410000
   0x000000000000000d (FINI)               0x5671b4
   0x0000000000000019 (INIT_ARRAY)         0x5d4c58
   0x000000000000001b (INIT_ARRAYSZ)       32 (bytes)
   0x000000000000001a (FINI_ARRAY)         0x5d4c78
   0x000000000000001c (FINI_ARRAYSZ)       8 (bytes)
   0x000000006ffffef5 (GNU_HASH)           0x400308
   0x0000000000000005 (STRTAB)             0x4024e0
   0x0000000000000006 (SYMTAB)             0x400590
   0x000000000000000a (STRSZ)              13777 (bytes)
   0x000000000000000b (SYMENT)             24 (bytes)
   0x0000000000000015 (DEBUG)              0x0
   0x0000000000000003 (PLTGOT)             0x5dffe8
   0x0000000000000002 (PLTRELSZ)           5328 (bytes)
   0x0000000000000014 (PLTREL)             RELA
   0x0000000000000017 (JMPREL)             0x408da8
   0x0000000000000007 (RELA)               0x405f10
   0x0000000000000008 (RELASZ)             11928 (bytes)
   0x0000000000000009 (RELAENT)            24 (bytes)
   0x000000006ffffffe (VERNEED)            0x405d50
   0x000000006fffffff (VERNEEDNUM)         6
   0x000000006ffffff0 (VERSYM)             0x405ab2
   0x0000000000000000 (NULL)               0x0

To check that you have configured it correctly, run::

  orangepi@orangepi5max:~$ ./sherpa-onnx-v1.12.13-rknn-linux-aarch64-shared/bin/sherpa-onnx --help

It should print help information for the binary ``sherpa-onnx``.

Build sherpa-onnx directly on your board
----------------------------------------

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa-onnx
   cd sherpa-onnx
   mkdir build
   cd build

   cmake \
     -DSHERPA_ONNX_ENABLE_RKNN=ON \
     -DCMAKE_INSTALL_PREFIX=./install \
     ..

   make
   make install

Cross-compiling
---------------

Please first refer to :ref:`sherpa-onnx-linux-aarch64-cross-compiling`
to install toolchains.

.. warning::

   The toolchains for dynamic linking and static linking are different.

After installing a toolchain by following :ref:`sherpa-onnx-linux-aarch64-cross-compiling`

Dynamic link
~~~~~~~~~~~~

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  export BUILD_SHARED_LIBS=ON
  ./build-rknn-linux-aarch64.sh

Static link
~~~~~~~~~~~

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  export BUILD_SHARED_LIBS=OFF
  ./build-rknn-linux-aarch64.sh
