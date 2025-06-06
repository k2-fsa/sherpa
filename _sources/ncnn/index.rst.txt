sherpa-ncnn
===========

.. hint::

  During speech recognition, it does not need to access the Internet.
  Everyting is processed locally on your device.


We support using `ncnn`_ to replace PyTorch for neural network computation.
The code is put in a separate repository `sherpa-ncnn`_

`sherpa-ncnn`_ is self-contained and everything can be compiled from source.

Please refer to `<https://k2-fsa.github.io/icefall/model-export/export-ncnn.html>`_
for how to export models to `ncnn`_ format.

In the following, we describe how to build `sherpa-ncnn`_ for Linux, macOS,
Windows, embedded systems, Android, and iOS.

Also, we show how to use it for speech recognition with pre-trained models.

.. toctree::
   :maxdepth: 2

   ./tutorials/index
   ./install/index
   ./python/index
   ./wasm/index
   ./c-api/index
   ./endpoint
   ./android/index
   ./ios/index
   ./pretrained_models/index
   ./examples/index
   ./faq
