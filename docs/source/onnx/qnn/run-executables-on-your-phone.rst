.. _run-exe-on-your-phone-with-qnn:

Run executables on your phone with adb (using libmodel.so)
==========================================================

In :ref:`build-sherpa-onnx-for-qualcomm-npu`, we have described how to generate
executable files. This section describes how to run them with QNN models (``libmodel.so``) on your
phone with adb.

.. hint::

   ``libmodel.so`` is **OS-dependent**, **QNN-SDK-independent**, but **SoC-independent**.

    - **OS-dependent**: a ``libmodel.so`` built for Android/arm64 cannot run on Linux/arm64, and vice-versa.
    - **QNN-SDK-independent**: Once built, ``libmodel.so`` does not depend on the version of the QNN SDK installed on the target device.
    - **SoC-independent**: the same ``libmodel.so`` can run on multiple Qualcomm chips such as SM8850, SA8259, QCS9100, and others.

   The trade-off is that the first-run initialization is slow, because the context has to be generated at runtime.

   If you want faster startup, use an SoC-specific but OS-independent context binary (``model.bin``).
   For guidance, see :ref:`run-exe-on-your-phone-with-qnn-binary`.

   .. include:: qnn_model_comparison.rst

Download a QNN model
--------------------

You can find available QNN models (``libmodel.so``) at

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn>`_

.. hint::

   For ``model.bin``, please see

    `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn-binary>`_

Since QNN does not support dynamic input shapes, we limit the maximum duration the model can handle.
For example, if the limit is 10 seconds, any input shorter than 10 seconds will be padded to 10 seconds,
and inputs longer than 10 seconds will be truncated to that length.

The model name indicates the maximum duration the model can handle.

.. caution::

   Please select a model with name ``-android-aarch64``.

We use ``sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64.tar.bz2``
as an example below:

.. code-block::

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models-qnn/sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64.tar.bz2

   tar xvf sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64.tar.bz2
   rm sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64.tar.bz2

You should see the following files:

.. code-block::

  ls -lh sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/
  total 492592
  -rw-r--r--@ 1 fangjun  staff    38B 10 Nov 11:03 info.txt
  -rwxr-xr-x@ 1 fangjun  staff   227M 10 Nov 11:03 libmodel.so
  -rw-r--r--@ 1 fangjun  staff    71B 10 Nov 11:03 LICENSE
  -rw-r--r--@ 1 fangjun  staff   104B 10 Nov 11:03 README.md
  drwxr-xr-x@ 7 fangjun  staff   224B 21 Nov 16:42 test_wavs
  -rw-r--r--@ 1 fangjun  staff   308K 10 Nov 11:03 tokens.txt

Copy files to your phone
------------------------

We assume you put files in the directory ``/data/local/tmp/sherpa-onnx`` on your phone.

.. code-block::

   # Run on your computer

   adb shell mkdir /data/local/tmp/sherpa-onnx

Copy model files
::::::::::::::::

.. code-block::

  # Run on your computer

  adb push ./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64 /data/local/tmp/sherpa-onnx/

Copy sherpa-onnx executable files
:::::::::::::::::::::::::::::::::

.. code-block::

  # Run on your computer

  adb push ./build-android-arm64-v8a/install/bin/sherpa-onnx-offline /data/local/tmp/sherpa-onnx/

Copy sherpa-onnx library files
::::::::::::::::::::::::::::::

.. code-block::

  # Run on your computer

  adb push ./build-android-arm64-v8a/install/lib/libonnxruntime.so /data/local/tmp/sherpa-onnx/

.. hint::

   You don't need to copy ``libsherpa-onnx-jni.so`` in this case.


Copy QNN library files
::::::::::::::::::::::

Before you continue, we assume you have followed :ref:`onnx-download-qnn`
to download QNN SDK and set up the environment variable ``QNN_SDK_ROOT``.

You should run::

  echo $QNN_SDK_ROOT

to check that it points to the QNN SDK directory.

.. code-block::

  # Run on your computer

  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp/sherpa-onnx/

  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV68Stub.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV69Stub.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV73Stub.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV79Stub.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so /data/local/tmp/sherpa-onnx/

  adb push $QNN_SDK_ROOT/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so /data/local/tmp/sherpa-onnx/
  adb push $QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so /data/local/tmp/sherpa-onnx/

.. hint::

   To make things easier, we have copied many unused Stub and Skel libraries. For a given
   device, you only need one Stub and one Skel library.

   For instance, if you are using Xiaomi 17 Pro, you only need to copy ``libQnnHtpV81Stub.so``
   and ``libQnnHtpV81Skel.so`` to your phone.

Run it !
--------

.. code-block::

   adb shell

The following commands are run on your phone.

Check files
:::::::::::

First, check that you have followed above commands to copy files:

.. image:: ./pic/qnn-adb-files.jpg
   :align: center
   :alt: screenshot of expected files on your phone
   :width: 600

.. image:: ./pic/model-files.png
   :align: center
   :alt: screenshot of expected models files on your phone
   :width: 600

Set environment variable ADSP_LIBRARY_PATH
::::::::::::::::::::::::::::::::::::::::::

.. code-block::

   export ADSP_LIBRARY_PATH="$PWD;$ADSP_LIBRARY_PATH"

where ``$PWD`` is ``/data/local/tmp/sherpa-onnx`` in this case.

.. caution::

   Please use ``;``, not ``:``.

   It is an ``error`` to use ``export ADSP_LIBRARY_PATH="$PWD:$ADSP_LIBRARY_PATH"``

   It is an ``error`` to use ``export ADSP_LIBRARY_PATH="$PWD:$ADSP_LIBRARY_PATH"``

   It is an ``error`` to use ``export ADSP_LIBRARY_PATH="$PWD:$ADSP_LIBRARY_PATH"``

.. image:: ./pic/adsp-path.jpg
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
     --sense-voice-model=./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/libmodel.so \
     --tokens=./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/tokens.txt \
     --sense-voice.qnn-backend-lib=./libQnnHtp.so \
     --sense-voice.qnn-system-lib=./libQnnSystem.so \
     --sense-voice.qnn-context-binary=./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/model.bin \
     ./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/test_wavs/zh.wav

or write it in a single line:

.. code-block::

   ./sherpa-onnx-offline --provider=qnn --sense-voice-model=./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/libmodel.so --tokens=./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/tokens.txt --sense-voice.qnn-backend-lib=./libQnnHtp.so --sense-voice.qnn-system-lib=./libQnnSystem.so --sense-voice.qnn-context-binary=./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/model.bin ./sherpa-onnx-qnn-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64/test_wavs/zh.wav


We give the logs of the first run and the second run. You can see that the initialization time for the
second run is significantly faster than that of the first run.

You can also find that the first run generates the following file (``model.bin``):

.. image:: ./pic/qnn-model-bin.jpg
   :align: center
   :alt: screenshot of ``model.bin``.
   :width: 600

Log of the first run
~~~~~~~~~~~~~~~~~~~~

.. container:: toggle

    .. container:: header

      Click ▶ to see the log of the 1st run.

    .. literalinclude:: ./code/1st-run.txt

    Please ignore the ``num_threads`` information in the log. It is not used by qnn.

    You can see it takes about 20 seconds to initialize the recognizer.
    It happens only in the 1st run. Subsequent runs are significantly faster.


Log of later runs
~~~~~~~~~~~~~~~~~

.. container:: toggle

    .. container:: header

      Click ▶ to see the log of the 2nd run.

    .. literalinclude:: ./code/2nd-run.txt

    Please ignore the ``num_threads`` information in the log. It is not used by qnn.

    You can see it takes only about 1 second to initialize the recognizer.


Congratulations
---------------

Congratulations! You have successfully launched sherpa-onnx on your phone,
leveraging Qualcomm NPU via QNN with the HTP backend.
