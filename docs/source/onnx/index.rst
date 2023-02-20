sherpa-onnx
===========

We support using `onnx`_ with `onnxruntime`_ to replace `PyTorch`_ for neural
network computation. The code is put in a separate repository `sherpa-onnx`_.

`sherpa-onnx`_ is self-contained and everything can be compiled from source.

Please refer to
`<https://k2-fsa.github.io/icefall/model-export/export-onnx.html>`_
for how to export models to `onnx`_ format.

.. hint::

   We use pre-compiled `onnxruntime`_ from
   `<https://github.com/microsoft/onnxruntime/releases>`_.

.. toctree::
   :maxdepth: 2

   ./install/index
   ./python/index
   ./pretrained_models/index
