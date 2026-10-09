Speaker Diarization
===================

This page describes how to use `sherpa-onnx`_ for speaker diarization.

Pre-trained models for speaker segmentation can be found
at `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models>`_

`Nemotron-3-Diarization`_ (Sortformer) does all of the speaker diarization with one model.
It does not need a speaker embedding model or clustering. See
:ref:`sherpa-onnx-nemotron-3-diarization` for details.

.. _Nemotron-3-Diarization: https://huggingface.co/nvidia/Nemotron-3-Diarization

Pre-trained models for speaker embedding extraction can be found
at `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models>`_

In the following, we describe different programming language APIs for speaker diarization.

.. toctree::
   :maxdepth: 5

   ./models.rst
   ./hf.rst
   ./android.rst
   ./c.rst
   ./cpp.rst
   ./csharp.rst
   ./dart.rst
   ./go.rst
   ./java.rst
   ./javascript.rst
   ./kotlin.rst
   ./pascal.rst
   ./python.rst
   ./rust.rst
   ./swift.rst
