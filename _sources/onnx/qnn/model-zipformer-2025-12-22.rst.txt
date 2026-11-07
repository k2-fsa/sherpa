sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8 (Chinese)
----------------------------------------------------------------------------------

This model is converted from :ref:`sherpa-onnx-zipformer-ctc-zh-int8-2025-12-22`.

.. code-block::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models-qnn-binary/sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8.tar.bz2
  tar xvf sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8.tar.bz2
  rm sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8.tar.bz2

Now copy them to your Qualcomm device. Make sure you have read :ref:`run-exe-on-your-phone-with-qnn-binary`
to copy required ``*.so`` files from QNN SDK ``2.40.0.251030`` and setup the environment variable ``ADSP_LIBRARY_PATH``.

.. code-block::

  pandora:/data/local/tmp $ ls -lh
  total 104K
  -rw-rw-rw- 1 shell shell 2.3M 2025-11-20 11:05 libQnnHtp.so
  -rw-rw-rw- 1 shell shell  71M 2025-11-20 11:05 libQnnHtpPrepare.so
  -rw-rw-rw- 1 shell shell  10M 2025-11-20 11:10 libQnnHtpV81Skel.so
  -rw-rw-rw- 1 shell shell 618K 2025-11-20 11:06 libQnnHtpV81Stub.so
  -rw-rw-rw- 1 shell shell 2.4M 2025-11-20 11:06 libQnnSystem.so
  -rw-rw-rw- 1 shell shell  15M 2025-12-10 11:43 libonnxruntime.so
  -rwxrwxrwx 1 shell shell 2.1M 2025-12-22 17:31 sherpa-onnx-offline
  drwxrwxr-x 3 shell shell 3.3K 2025-12-22 17:45 sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8
  pandora:/data/local/tmp $ ls -lh sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8
  total 351K
  -rw-rw-rw- 1 shell shell   15 2025-12-22 17:29 info.txt
  -rw-rw-rw- 1 shell shell 351M 2025-12-22 17:29 model.bin
  drwxrwxr-x 2 shell shell 3.3K 2025-12-22 17:45 test_wavs
  -rw-rw-rw- 1 shell shell  13K 2025-12-22 17:29 tokens.txt
  pandora:/data/local/tmp $ export ADSP_LIBRARY_PATH="$PWD;$ADSP_LIBRARY_PATH"
  pandora:/data/local/tmp $


Run it on your device:

.. code-block::

   ./sherpa-onnx-offline \
     --provider=qnn \
     --tokens=./sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8/tokens.txt \
     --zipformer-ctc.qnn-backend-lib=./libQnnHtp.so \
     --zipformer-ctc.qnn-system-lib=./libQnnSystem.so \
     --zipformer-ctc.qnn-context-binary=./sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8/model.bin \
     ./sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8/test_wavs/0.wav

or write it in a single line:

.. code-block::

   ./sherpa-onnx-offline --provider=qnn --tokens=./sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8/tokens.txt --zipformer-ctc.qnn-backend-lib=./libQnnHtp.so --zipformer-ctc.qnn-system-lib=./libQnnSystem.so --zipformer-ctc.qnn-context-binary=./sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8/model.bin ./sherpa-onnx-qnn-SM8850-binary-5-seconds-zipformer-ctc-zh-2025-12-22-int8/test_wavs/0.wav

You can find the output log below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the log .

    .. literalinclude:: ./code/zipformer-8850-5-seconds-binary-2025-12-22.txt

    Please ignore the ``num_threads`` information in the log. It is not used by qnn.

    .. hint::

       The model actually processed only ``5`` seconds of audio, not ``5.592`` seconds.


