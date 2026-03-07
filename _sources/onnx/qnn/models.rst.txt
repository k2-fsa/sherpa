Models for QNN
==============

We assume you have already read :ref:`run-exe-on-your-phone-with-qnn-binary`
and are familiar with setting up the environment.

This section provides a list of the models supported by QNN in `sherpa-onnx`_.

It only shows the usage of models from

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn-binary>`_

.. hint::

   Please see :ref:`run-exe-on-your-phone-with-qnn` for models from

    `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models-qnn>`_

.. caution::

   - I am using a Xiaomi 17 Pro for testing, so I selected a model with SM8850 in its name in this section.
   - Make sure to select a model that matches your own device.
   - Suppose you are testing on a Samsung Galaxy S23 Ultra, which uses the SM8550 SoC;
     In this case, you should select a model with SM8550 in its name instead of SM8850.

Since QNN does not support dynamic input shapes, we limit the maximum duration the model can handle.
For example, if the limit is 10 seconds, any input shorter than 10 seconds will be padded to 10 seconds,
and inputs longer than 10 seconds will be truncated to that length.

The model name indicates the maximum duration the model can handle. We use ``5-seconds``
in this section as an example.


.. include:: ./model-zipformer-2025-12-22.rst

.. include:: ./model-zipformer-2025-07-03.rst

.. include:: ./model-sense-voice-2024-07-17.rst
