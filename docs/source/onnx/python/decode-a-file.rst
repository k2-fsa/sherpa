Decode a file
=============

In this section, we demonstrate how to use the Python API of `sherpa-onnx`_
to recognize a file.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   python3 ./python-api-examples/decode-file.py \
     --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
     --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
     --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
     --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
     --wave-filename=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/4.wav

.. hint::

   ``decode-file.py`` is from `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/decode-file.py>`_

   In the above demo, the model files are
   from :ref:`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`.

The output is given below:

.. code-block:: bash

  Started!
  嗯 ON TIME要准时 IN TIME是及时叫他总是准时教他的作业那用一般现在时是没有什么
  感情色彩的陈述一个事实下一句话为什么要用现在进行时它的意思并不是说说他现在正
  在教他的
  Done!
  num_threads: 2
  Wave duration: 17.640 s
  Elapsed time: 1.504 s
  Real time factor (RTF): 1.504/17.640 = 0.085

