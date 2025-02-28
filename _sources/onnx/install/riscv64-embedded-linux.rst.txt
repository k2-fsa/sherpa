Embedded Linux (riscv64)
========================

This page describes how to build `sherpa-onnx`_ for embedded Linux (RISC-V, 64-bit)
with cross-compiling on an x64 machine with Ubuntu OS. It also demonstrates
how to use ``qemu`` to run the compiled binaries.

.. hint::

   We provide a colab notebook
   |build sherpa-onnx for risc-v colab notebook|
   for you to try this section step by step.

   If you are using Windows/macOS or you don't want to setup your local environment
   for cross-compiling, please use the above colab notebook.

.. |build sherpa-onnx for risc-v colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_RISC_V.ipynb

.. note::

   You can download pre-compiled binaries for ``riscv64`` from the following URL
   `<https://huggingface.co/csukuangfj/sherpa-onnx-libs/tree/main/riscv64>`_

   Please always download the latest version.

   Example command to download the version ``1.9.12``:

    .. code-block:: bash

      # binaries built with shared libraries
      wget https://huggingface.co/csukuangfj/sherpa-onnx-libs/resolve/main/riscv64/sherpa-onnx-v1.9.12-linux-riscv64-shared.tar.bz2

      # For users from China
      # 中国国内用户，如果访问不了 huggingface, 请使用

      # binaries built with shared libraries
      # wget https://hf-mirror.com/csukuangfj/sherpa-onnx-libs/resolve/main/riscv64/sherpa-onnx-v1.9.12-linux-riscv64-shared.tar.bz2

Install toolchain
-----------------

The first step is to install a toolchain for cross-compiling.

.. code-block:: bash

   mkdir -p $HOME/toolchain

   wget -q https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1663142514282/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz

   tar xf ./Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz --strip-components 1 -C $HOME/toolchain

Next, we need to set the following environment variable:

.. code-block:: bash

   export PATH=$HOME/toolchain/bin:$PATH

To check that you have installed the toolchain successfully, please run

.. code-block:: bash

  $ riscv64-unknown-linux-gnu-gcc --version

    riscv64-unknown-linux-gnu-gcc (Xuantie-900 linux-5.10.4 glibc gcc Toolchain V2.6.1 B-20220906) 10.2.0
    Copyright (C) 2020 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  $ riscv64-unknown-linux-gnu-g++ --version

    riscv64-unknown-linux-gnu-g++ (Xuantie-900 linux-5.10.4 glibc gcc Toolchain V2.6.1 B-20220906) 10.2.0
    Copyright (C) 2020 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Build sherpa-onnx
-----------------

Next, let us build `sherpa-onnx`_.

.. hint::

   Currently, only shared libraries are supported. We ``will`` support
   static linking in the future.

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx
  ./build-riscv64-linux-gnu.sh

After building, you will get the following files

.. code-block:: bash

  $ ls -lh build-riscv64-linux-gnu/install/bin
  $ echo "---"
  $ ls -lh build-riscv64-linux-gnu/install/lib

    total 292K
    -rwxr-xr-x 1 root root 23K Mar 20 09:41 sherpa-onnx
    -rwxr-xr-x 1 root root 27K Mar 20 09:41 sherpa-onnx-alsa
    -rwxr-xr-x 1 root root 31K Mar 20 09:41 sherpa-onnx-alsa-offline
    -rwxr-xr-x 1 root root 40K Mar 20 09:41 sherpa-onnx-alsa-offline-speaker-identification
    -rwxr-xr-x 1 root root 23K Mar 20 09:41 sherpa-onnx-keyword-spotter
    -rwxr-xr-x 1 root root 27K Mar 20 09:41 sherpa-onnx-keyword-spotter-alsa
    -rwxr-xr-x 1 root root 23K Mar 20 09:41 sherpa-onnx-offline
    -rwxr-xr-x 1 root root 39K Mar 20 09:41 sherpa-onnx-offline-parallel
    -rwxr-xr-x 1 root root 19K Mar 20 09:41 sherpa-onnx-offline-tts
    -rwxr-xr-x 1 root root 31K Mar 20 09:41 sherpa-onnx-offline-tts-play-alsa
    ---
    total 30M
    -rw-r--r-- 1 root root 256K Mar 20 09:41 libespeak-ng.so
    -rw-r--r-- 1 root root  71K Mar 20 09:41 libkaldi-decoder-core.so
    -rw-r--r-- 1 root root  67K Mar 20 09:41 libkaldi-native-fbank-core.so
    -rw-r--r-- 1 root root  13M Mar 20 09:35 libonnxruntime.so
    -rw-r--r-- 1 root root  13M Mar 20 09:35 libonnxruntime.so.1.14.1
    lrwxrwxrwx 1 root root   23 Mar 20 09:41 libpiper_phonemize.so -> libpiper_phonemize.so.1
    lrwxrwxrwx 1 root root   27 Mar 20 09:41 libpiper_phonemize.so.1 -> libpiper_phonemize.so.1.2.0
    -rw-r--r-- 1 root root 395K Mar 20 09:41 libpiper_phonemize.so.1.2.0
    -rw-r--r-- 1 root root 1.3M Mar 20 09:41 libsherpa-onnx-core.so
    lrwxrwxrwx 1 root root   23 Mar 20 09:41 libsherpa-onnx-fst.so -> libsherpa-onnx-fst.so.6
    -rw-r--r-- 1 root root 1.4M Mar 20 09:41 libsherpa-onnx-fst.so.6
    -rw-r--r-- 1 root root 752K Mar 20 09:41 libsherpa-onnx-kaldifst-core.so
    -rw-r--r-- 1 root root 202K Mar 20 09:41 libucd.so
    drwxr-xr-x 2 root root 4.0K Mar 20 09:41 pkgconfig

.. code-block:: bash

   $ file build-riscv64-linux-gnu/install/bin/sherpa-onnx

   build-riscv64-linux-gnu/install/bin/sherpa-onnx: ELF 64-bit LSB executable, UCB RISC-V, RVC, double-float ABI, version 1 (GNU/Linux), dynamically linked, interpreter /lib/ld-linux-riscv64-lp64d.so.1, for GNU/Linux 4.15.0, stripped

.. code-block:: bash

   $ readelf -d build-riscv64-linux-gnu/install/bin/sherpa-onnx

.. code-block:: bash

   $ find $HOME/toolchain/ -name ld-linux-riscv64-lp64d.so.1

      Dynamic section at offset 0x4d40 contains 39 entries:
        Tag        Type                         Name/Value
       0x0000000000000001 (NEEDED)             Shared library: [libsherpa-onnx-core.so]
       0x0000000000000001 (NEEDED)             Shared library: [libkaldi-native-fbank-core.so]
       0x0000000000000001 (NEEDED)             Shared library: [libkaldi-decoder-core.so]
       0x0000000000000001 (NEEDED)             Shared library: [libsherpa-onnx-kaldifst-core.so]
       0x0000000000000001 (NEEDED)             Shared library: [libsherpa-onnx-fst.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libpiper_phonemize.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libonnxruntime.so.1.14.1]
       0x0000000000000001 (NEEDED)             Shared library: [libespeak-ng.so]
       0x0000000000000001 (NEEDED)             Shared library: [libucd.so]
       0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
       0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
       0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
       0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
       0x000000000000000f (RPATH)              Library rpath: [$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../../sherpa_onnx/lib]
       0x0000000000000020 (PREINIT_ARRAY)      0x15d20
       0x0000000000000021 (PREINIT_ARRAYSZ)    8 (bytes)
       0x0000000000000019 (INIT_ARRAY)         0x15d28
       0x000000000000001b (INIT_ARRAYSZ)       16 (bytes)
       0x000000000000001a (FINI_ARRAY)         0x15d38
       0x000000000000001c (FINI_ARRAYSZ)       8 (bytes)
       0x0000000000000004 (HASH)               0x10280
       0x000000006ffffef5 (GNU_HASH)           0x10418
       0x0000000000000005 (STRTAB)             0x10bd8
       0x0000000000000006 (SYMTAB)             0x105f0
       0x000000000000000a (STRSZ)              3652 (bytes)
       0x000000000000000b (SYMENT)             24 (bytes)
       0x0000000000000015 (DEBUG)              0x0
       0x0000000000000003 (PLTGOT)             0x16000
       0x0000000000000002 (PLTRELSZ)           1056 (bytes)
       0x0000000000000014 (PLTREL)             RELA
       0x0000000000000017 (JMPREL)             0x11bb0
       0x0000000000000007 (RELA)               0x11b80
       0x0000000000000008 (RELASZ)             1104 (bytes)
       0x0000000000000009 (RELAENT)            24 (bytes)
       0x000000006ffffffe (VERNEED)            0x11aa0
       0x000000006fffffff (VERNEEDNUM)         4
       0x000000006ffffff0 (VERSYM)             0x11a1c
       0x0000000000000000 (NULL)               0x0

    /root/toolchain/sysroot/lib/ld-linux-riscv64-lp64d.so.1


That's it!

Please create an issue at `<https://github.com/k2-fsa/sherpa-onnx/issues>`_
if you have any problems.

Read more if you want to run the binaries with ``qemu``.

qemu
----

.. hint::

   This subsection works only on x64 Linux.

.. caution::

   Please don't use any other methods to install ``qemu-riscv64``. Only the
   method listed in this subsection is known to work.

Please use the following command to download the ``qemu-riscv64`` binary.

.. code-block:: bash

   mkdir -p $HOME/qemu

   mkdir -p /tmp
   cd /tmp
   wget -q https://files.pythonhosted.org/packages/21/f4/733f29c435987e8bb264a6504c7a4ea4c04d0d431b38a818ab63eef082b9/xuantie_qemu-20230825-py3-none-manylinux1_x86_64.whl

   unzip xuantie_qemu-20230825-py3-none-manylinux1_x86_64.whl
   cp -v ./qemu/qemu-riscv64 $HOME/qemu

   export PATH=$HOME/qemu:$PATH

To check that we have installed ``qemu-riscv64`` successfully, please run:

.. code-block:: bash

    qemu-riscv64 -h

which should give the following output::

    usage: qemu-riscv64 [options] program [arguments...]
    Linux CPU emulator (compiled for riscv64 emulation)

    Options and associated environment variables:

    Argument             Env-variable      Description
    -h                                     print this help
    -help
    -g port              QEMU_GDB          wait gdb connection to 'port'
    -L path              QEMU_LD_PREFIX    set the elf interpreter prefix to 'path'
    -s size              QEMU_STACK_SIZE   set the stack size to 'size' bytes
    -cpu model           QEMU_CPU          select CPU (-cpu help for list)
    -E var=value         QEMU_SET_ENV      sets targets environment variable (see below)
    -U var               QEMU_UNSET_ENV    unsets targets environment variable (see below)
    -0 argv0             QEMU_ARGV0        forces target process argv[0] to be 'argv0'
    -r uname             QEMU_UNAME        set qemu uname release string to 'uname'
    -B address           QEMU_GUEST_BASE   set guest_base address to 'address'
    -R size              QEMU_RESERVED_VA  reserve 'size' bytes for guest virtual address space
    -d item[,...]        QEMU_LOG          enable logging of specified items (use '-d help' for a list of items)
    -dfilter range[,...] QEMU_DFILTER      filter logging based on address range
    -D logfile           QEMU_LOG_FILENAME write logs to 'logfile' (default stderr)
    -p pagesize          QEMU_PAGESIZE     set the host page size to 'pagesize'
    -singlestep          QEMU_SINGLESTEP   run in singlestep mode
    -strace              QEMU_STRACE       log system calls
    -pctrace             QEMU_PCTRACE      log pctrace
    -seed                QEMU_RAND_SEED    Seed for pseudo-random number generator
    -trace               QEMU_TRACE        [[enable=]<pattern>][,events=<file>][,file=<file>]
    -csky-extend         CSKY_EXTEND       [tb_trace=<on|off>][,jcount_start=<addr>][,jcount_end=<addr>][vdsp=<vdsp>][exit_addr=<addr>][denormal=<on|off>]
    -CPF                 CSKY_PROFILING
    -csky-trace          CSKY_TRACE        [port=<port>][,tb_trace=<on|off>][,mem_trace=<on|off>][,auto_trace=<on|off>][,start=addr][,exit=addr]
    -plugin              QEMU_PLUGIN       [file=]<file>[,arg=<string>]
    -version             QEMU_VERSION      display version information and exit

    Defaults:
    QEMU_LD_PREFIX  = /usr/gnemul/qemu-riscv64
    QEMU_STACK_SIZE = 8388608 byte

    You can use -E and -U options or the QEMU_SET_ENV and
    QEMU_UNSET_ENV environment variables to set and unset
    environment variables for the target process.
    It is possible to provide several variables by separating them
    by commas in getsubopt(3) style. Additionally it is possible to
    provide the -E and -U options multiple times.
    The following lines are equivalent:
        -E var1=val2 -E var2=val2 -U LD_PRELOAD -U LD_DEBUG
        -E var1=val2,var2=val2 -U LD_PRELOAD,LD_DEBUG
        QEMU_SET_ENV=var1=val2,var2=val2 QEMU_UNSET_ENV=LD_PRELOAD,LD_DEBUG
    Note that if you provide several changes to a single variable
    the last change will stay in effect.

    See <https://qemu.org/contribute/report-a-bug> for how to report bugs.
    More information on the QEMU project at <https://qemu.org>.

We describe below how to use ``qemu-riscv64`` to run speech-to-text and text-to-speech.


Run speech-to-text with qemu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use :ref:`sherpa_onnx_streaming_zipformer_en_20M_2023_02_17` as the test model.

.. note::

   You can select any model from :ref:`sherpa-onnx-pre-trained-models`.


Please use the following command to download the model:

.. code-block:: bash

    cd /path/to/sherpa-onnx

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
    tar xvf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
    rm sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

Now you can use the following command to run it with ``qemu-riscv64``::

  cd /path/to/sherpa-onnx

  export PATH=$HOME/qemu:$PATH

  qemu-riscv64 build-riscv64-linux-gnu/install/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav

It will throw the following error::

  qemu-riscv64: Could not open '/lib/ld-linux-riscv64-lp64d.so.1': No such file or directory

Please use the following command instead::

  cd /path/to/sherpa-onnx

  export PATH=$HOME/qemu:$PATH
  export QEMU_LD_PREFIX=$HOME/toolchain/sysroot

  qemu-riscv64 build-riscv64-linux-gnu/install/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav

It will throw a second error::

  build-riscv64-linux-gnu/install/bin/sherpa-onnx: error while loading shared libraries: ld-linux-riscv64xthead-lp64d.so.1: cannot open shared object file: No such file or directory

Please use the following command instead::

  cd /path/to/sherpa-onnx

  export PATH=$HOME/qemu:$PATH
  export QEMU_LD_PREFIX=$HOME/toolchain/sysroot
  export LD_LIBRARY_PATH=$HOME/toolchain/sysroot/lib:$LD_LIBRARY_PATH

  qemu-riscv64 build-riscv64-linux-gnu/install/bin/sherpa-onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt \
    --encoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx \
    --decoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx \
    --joiner=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx \
    ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav

Finally, it prints the following output::

  /content/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:361 build-riscv64-linux-gnu/install/bin/sherpa-onnx --tokens=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt --encoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx --decoder=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx --joiner=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav

  OnlineRecognizerConfig(feat_config=FeatureExtractorConfig(sampling_rate=16000, feature_dim=80), model_config=OnlineModelConfig(transducer=OnlineTransducerModelConfig(encoder="./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx", decoder="./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx", joiner="./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx"), paraformer=OnlineParaformerModelConfig(encoder="", decoder=""), wenet_ctc=OnlineWenetCtcModelConfig(model="", chunk_size=16, num_left_chunks=4), zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=""), tokens="./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt", num_threads=1, debug=False, provider="cpu", model_type=""), lm_config=OnlineLMConfig(model="", scale=0.5), endpoint_config=EndpointConfig(rule1=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=2.4, min_utterance_length=0), rule2=EndpointRule(must_contain_nonsilence=True, min_trailing_silence=1.2, min_utterance_length=0), rule3=EndpointRule(must_contain_nonsilence=False, min_trailing_silence=0, min_utterance_length=20)), enable_endpoint=True, max_active_paths=4, hotwords_score=1.5, hotwords_file="", decoding_method="greedy_search", blank_penalty=0)
  ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav
  Elapsed seconds: 70, Real time factor (RTF): 11
   THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BRAFFLELS
  { "text": " THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BRAFFLELS", "tokens": [ " THE", " YE", "LL", "OW", " LA", "M", "P", "S", " WOULD", " LIGHT", " UP", " HE", "RE", " AND", " THERE", " THE", " S", "QUA", "LI", "D", " ", "QUA", "R", "TER", " OF", " THE", " B", "RA", "FF", "L", "EL", "S" ], "timestamps": [ 2.04, 2.16, 2.28, 2.36, 2.52, 2.64, 2.68, 2.76, 2.92, 3.08, 3.40, 3.60, 3.72, 3.88, 4.12, 4.48, 4.64, 4.68, 4.84, 4.96, 5.16, 5.20, 5.32, 5.36, 5.60, 5.72, 5.92, 5.96, 6.08, 6.24, 6.36, 6.60 ], "ys_probs": [ -0.454799, -0.521409, -0.345871, -0.001244, -0.240359, -0.013972, -0.010445, -0.051701, -0.000371, -0.171570, -0.002205, -0.026703, -0.006903, -0.021168, -0.011662, -0.001059, -0.005089, -0.000273, -0.575480, -0.024973, -0.159344, -0.000042, -0.011082, -0.187136, -0.004002, -0.292751, -0.084873, -0.241302, -0.543844, -0.428164, -0.853198, -0.093776 ], "lm_probs": [  ], "context_scores": [  ], "segment": 0, "start_time": 0.00, "is_final": false}

.. hint::

   As you can see, the RTF is 11, indicating that it is very slow to run the model
   with the ``qemu`` simulator. Running on a real RISC-V board should be much faster.

Run text-to-speech with qemu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please visit `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_
to download a text-to-speech model. We use the following model
``vits-piper-en_US-amy-low.tar.bz2``::

  cd /path/to/sherpa-onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2
  rm vits-piper-en_US-amy-low.tar.bz2

After downloading the model, we can use the following command to run it::

  cd /path/to/sherpa-onnx

  export PATH=$HOME/qemu:$PATH
  export QEMU_LD_PREFIX=$HOME/toolchain/sysroot
  export LD_LIBRARY_PATH=$HOME/toolchain/sysroot/lib:$LD_LIBRARY_PATH

  qemu-riscv64 build-riscv64-linux-gnu/install/bin/sherpa-onnx-offline-tts \
    --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
    --vits-tokens=./vits-piper-en_US-amy-low/tokens.txt \
    --vits-data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
    --output-filename=./a-test.wav \
    "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

The log of the above command is given below::

  /content/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:361 build-riscv64-linux-gnu/install/bin/sherpa-onnx-offline-tts --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx --vits-tokens=./vits-piper-en_US-amy-low/tokens.txt --vits-data-dir=./vits-piper-en_US-amy-low/espeak-ng-data --output-filename=./a-test.wav 'Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone.'

  Elapsed seconds: 270.745 s
  Audio duration: 7.904 s
  Real-time factor (RTF): 270.745/7.904 = 34.254
  The text is: Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone.. Speaker ID: 0
  Saved to ./a-test.wav successfully!

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>a-test.wav</td>
      <td>
       <audio title="Generated a-test.wav" controls="controls">
             <source src="/sherpa/_static/onnx/riscv64/a-test.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone.
      </td>
    </tr>
  </table>
