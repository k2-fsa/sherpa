How to export models from WeNet to sherpa-onnx
==============================================

Suppose you have the following files from `WeNet`_:

  - ``final.pt``
  - ``train.yaml``
  - ``global_cmvn``
  - ``units.txt``

We describe below how to use scripts from `sherpa-onnx`_ to export your files.

.. hint::

   Both streaming and non-streaming models are supported.

Export for non-streaming inference
----------------------------------

You can use the following script

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wenet/export-onnx.py>`_

to export your model to `sherpa-onnx`_. After running it, you should get two files:

  - ``model.onnx``
  - ``model.int8.onnx``.

Next, we rename ``units.txt`` to ``tokens.txt`` to follow the convention used in `sherpa-onnx`_:

.. code-block:: bash

    mv units.txt tokens.txt

Now you can use the following command for speech recognition with the exported models:

.. code-block:: bash

  # with float32 models
  ./build/bin/sherpa-onnx-offline \
    --wenet-ctc-model=./model.onnx
    --tokens=./tokens.txt \
    /path/to/some.wav

  # with int8 models
  ./build/bin/sherpa-onnx-offline \
    --wenet-ctc-model=./model.int8.onnx
    --tokens=./tokens.txt \
    /path/to/some.wav

Export for streaming inference
------------------------------

You can use the following script

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wenet/export-onnx-streaming.py>`_

to export your model to `sherpa-onnx`_. After running it, you should get two files:

  - ``model-streaming.onnx``
  - ``model-streaming.int8.onnx``.

Next, we rename ``units.txt`` to ``tokens.txt`` to follow the convention used in `sherpa-onnx`_:

.. code-block:: bash

    mv units.txt tokens.txt

Now you can use the following command for speech recognition with the exported models:

.. code-block:: bash

  # with float32 models
  ./build/bin/sherpa-onnx \
    --wenet-ctc-model=./model-streaming.onnx
    --tokens=./tokens.txt \
    /path/to/some.wav

  # with int8 models
  ./build/bin/sherpa-onnx \
    --wenet-ctc-model=./model-streaming.int8.onnx
    --tokens=./tokens.txt \
    /path/to/some.wav

FAQs
----

sherpa-onnx/csrc/online-wenet-ctc-model.cc:Init:144 head does not exist in the metadata
---------------------------------------------------------------------------------------

.. code-block::

   /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/online-wenet-ctc-model.cc:Init:144 head does not exist in the metadata

To fix the above error, please check the following two items:

  - Make sure you are using ``model-streaming.onnx`` or ``model-streaing.int8.onnx``. The executable
    you are running requires a streaming model as input.
  - Make sure you use the script from `sherpa-onnx`_ to export your model.
