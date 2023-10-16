sherpa-mnn
===========

.. hint::

  We wrote this section of the document by referring to `sherpa-ncnn`_ and made
  only necessary modifications.

The model needs one inference engine during deployment, and we have provided
 `sherpa-ncnn`_ based on the ncnn inference engine. This section introduces
 `sherpa-mnn`_ which is based on another inference engine, MNN.


`sherpa-mnn`_ is self-contained and everything can be compiled from source.

Please refer to `<https://k2-fsa.github.io/icefall/model-export/export-mnn.html>`_
for how to export models to `MNN`_ format.

In the following, we describe how to build `sherpa-mnn`_ for Linux, and embedded
systems. Other systems (macOS, Windows, Android, and iOS) are currently untested
, please refer to `sherpa-ncnn`_.

Also, we show how to use it for speech recognition with pre-trained models.

.. toctree::
   :maxdepth: 2

   ./install/index
   ./pretrained_models/index
