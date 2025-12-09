.. _run-exe-on-your-phone-with-qnn-binary:

Run executables on your phone with adb (using model.bin)
========================================================

In :ref:`build-sherpa-onnx-for-qualcomm-npu`, we have described how to generate
executable files. This section describes how to run them with QNN models (``model.bin``) on your
phone with adb.

.. hint::

    ``model.bin`` is **OS-independent**, but **QNN-SDK-dependent** and **SoC-dependent**.

      - **OS-independent**: A ``model.bin`` can run on both Android/arm64 and Linux/arm64.

      - **QNN-SDK-dependent**: Once built, ``model.bin`` depends on the version of the QNN SDK used during its creation.

      - **SoC-dependent**: A ``model.bin`` built for SM8850 cannot be used on SA8259, and vice versa.

      - **Trade-off**: The first-run initialization is extremely fast because the context is pre-generated.

      - **Alternative**: If you need **SoC-independence** or **QNN-SDK-independence**, use ``libmodel.so``.
        For guidance, see :ref:`run-exe-on-your-phone-with-qnn`.

    .. include:: qnn_model_comparison.rst

Download a QNN model
--------------------

You can find available QNN models at

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn-binary>`_

.. hint::

   For ``libmodel.so``, please see

    `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn>`_

Since QNN does not support dynamic input shapes, we limit the maximum duration the model can handle.
For example, if the limit is 10 seconds, any input shorter than 10 seconds will be padded to 10 seconds,
and inputs longer than 10 seconds will be truncated to that length.

The model name indicates the maximum duration the model can handle.

.. caution::

   - I am using a Xiaomi 17 Pro for testing, so I selected a model with SM8850 in its name.
   - Make sure to select a model that matches your own device.
   - Suppose you are testing on a Samsung Galaxy S23 Ultra, which uses the SM8550 SoC;
     In this case, you should select a model with SM8550 in its name instead of SM8850.

We use ``sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2``
as an example below:

.. code-block::

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models-qnn-binary/sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2
   tar xvf sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2
   rm sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8.tar.bz2

You should see the following files:

.. code-block::

  ls -lh sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/
  total 526984
  -rw-r--r--@ 1 fangjun  staff    23B  9 Dec 16:40 info.txt
  -rw-r--r--@ 1 fangjun  staff    71B  9 Dec 16:40 LICENSE
  -rw-r--r--@ 1 fangjun  staff   242M  9 Dec 16:40 model.bin
  -rw-r--r--@ 1 fangjun  staff   104B  9 Dec 16:40 README.md
  drwxr-xr-x@ 7 fangjun  staff   224B  9 Dec 16:40 test_wavs
  -rw-r--r--@ 1 fangjun  staff   308K  9 Dec 16:40 tokens.txt

Copy files to your phone
------------------------

We assume you put files in the directory ``/data/local/tmp/binary`` on your phone.

.. code-block::

   # Run on your computer

   adb shell mkdir /data/local/tmp/binary

Copy model files
::::::::::::::::

.. code-block::

  # Run on your computer

  adb push ./sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8 /data/local/tmp/binary/

Copy sherpa-onnx executable files
:::::::::::::::::::::::::::::::::

.. code-block::

  # Run on your computer

  adb push ./build-android-arm64-v8a/install/bin/sherpa-onnx-offline /data/local/tmp/binary/

Copy sherpa-onnx library files
::::::::::::::::::::::::::::::

.. code-block::

  # Run on your computer

  adb push ./build-android-arm64-v8a/install/lib/libonnxruntime.so /data/local/tmp/binary/

.. hint::

   You don't need to copy ``libsherpa-onnx-jni.so`` in this case.


Copy QNN library files
::::::::::::::::::::::

Before you continue, we assume you have followed :ref:`onnx-download-qnn`
to download QNN SDK and set up the environment variable ``QNN_SDK_ROOT``.

You should run::

  echo $QNN_SDK_ROOT

to check that it points to the QNN SDK directory.

.. warning::

   We use QNN SDK ``v2.40.0.251030`` to generate ``model.bin``.

   If you change the QNN SDK version, please re-generate the ``model.bin`` by yourself.

.. code-block::

  # Run on your computer

  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/binary/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so /data/local/tmp/binary/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp/binary/

  # Since my Xiami 17 Pro is SM8850, which corresponds to Htp Arch 81, so I choose
  # libQnnHtpV81Stub.so and libQnnHtpV81Skel.so
  #
  # Please udpate it accordingly to match your device
  #
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so /data/local/tmp/binary/

  adb push $QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so /data/local/tmp/binary/

Run it !
--------

.. code-block::

   adb shell

The following commands are run on your phone.

Check files
:::::::::::

First, check that you have followed above commands to copy files:

.. image:: ./pic/qnn-adb-files-binary.jpg
   :align: center
   :alt: screenshot of expected files on your phone
   :width: 600

.. image:: ./pic/model-files-binary.png
   :align: center
   :alt: screenshot of expected models files on your phone
   :width: 600

Set environment variable ADSP_LIBRARY_PATH
::::::::::::::::::::::::::::::::::::::::::

.. code-block::

   export ADSP_LIBRARY_PATH="$PWD;$ADSP_LIBRARY_PATH"

where ``$PWD`` is ``/data/local/tmp/binary`` in this case.

.. caution::

   Please use ``;``, not ``:``.

   It is an ``error`` to use ``export ADSP_LIBRARY_PATH="$PWD:$ADSP_LIBRARY_PATH"``

   It is an ``error`` to use ``export ADSP_LIBRARY_PATH="$PWD:$ADSP_LIBRARY_PATH"``

   It is an ``error`` to use ``export ADSP_LIBRARY_PATH="$PWD:$ADSP_LIBRARY_PATH"``

.. image:: ./pic/adsp-path-binary.jpg
   :align: center
   :alt: screenshot of setting ``ADSP_LIBRARY_PATH``
   :width: 600

Run sherpa-onnx-offline
:::::::::::::::::::::::

.. caution::

   You would be sad if you did not set the environment variable ``ADSP_LIBRARY_PATH``.

.. code-block::

   ./sherpa-onnx-offline \
     --provider=qnn \
     --tokens=./sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/tokens.txt \
     --sense-voice.qnn-backend-lib=./libQnnHtp.so \
     --sense-voice.qnn-system-lib=./libQnnSystem.so \
     --sense-voice.qnn-context-binary=./sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/model.bin \
     ./sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/test_wavs/zh.wav

or write it in a single line:

.. code-block::

   ./sherpa-onnx-offline --provider=qnn --tokens=./sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/tokens.txt --sense-voice.qnn-backend-lib=./libQnnHtp.so --sense-voice.qnn-system-lib=./libQnnSystem.so --sense-voice.qnn-context-binary=./sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/model.bin ./sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/test_wavs/zh.wav


You can also find the log below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the log .

    .. literalinclude:: ./code/binary-run.txt

    Please ignore the ``num_threads`` information in the log. It is not used by qnn.

.. hint::

   The model actually has processed 10 seconds of audio, so the RTF is even smaller.

Congratulations
---------------

Congratulations! You have successfully launched sherpa-onnx on your phone,
leveraging Qualcomm NPU via QNN with the HTP backend.
