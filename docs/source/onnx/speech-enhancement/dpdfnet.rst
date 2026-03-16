DPDFNet
=======

`DPDFNet <https://github.com/ceva-ip/DPDFNet>`_ is a family of causal,
single-channel speech enhancement models for real-time noise suppression.
It extends DeepFilterNet2 with Dual-Path RNN (DPRNN) blocks in the encoder
for stronger long-range temporal and cross-band modeling while staying
streaming-friendly. The paper is available on
`arXiv <https://arxiv.org/abs/2512.16420>`_, and the pre-trained model files
are hosted on `Hugging Face <https://huggingface.co/Ceva-IP/DPDFNet>`_.

In `sherpa-onnx`_, DPDFNet supports offline speech enhancement and online
streaming speech enhancement in the runtime and C API.

.. note::

   The current ``dpdfnet`` branch of `sherpa-onnx`_ downloads DPDFNet ONNX
   models from the official Hugging Face repository. Sample wave files such as
   ``inp_16k.wav`` and ``speech_with_noise.wav`` are hosted in the
   ``speech-enhancement-models`` GitHub release.

Model variants
--------------

.. list-table::
   :header-rows: 1

   * - Model
     - Params (M)
     - MACs (G)
     - Sample rate
     - Intended use
   * - ``baseline``
     - 2.31
     - 0.36
     - 16 kHz
     - Fastest / lowest resource usage
   * - ``dpdfnet2``
     - 2.49
     - 1.35
     - 16 kHz
     - Real-time / embedded devices
   * - ``dpdfnet4``
     - 2.84
     - 2.36
     - 16 kHz
     - Balanced performance
   * - ``dpdfnet8``
     - 3.54
     - 4.37
     - 16 kHz
     - Best enhancement quality
   * - ``dpdfnet2_48khz_hr``
     - 2.58
     - 2.42
     - 48 kHz
     - High-resolution audio

.. hint::

   Use ``baseline``, ``dpdfnet2``, ``dpdfnet4``, or ``dpdfnet8`` for 16 kHz
   downstream ASR or speech recognition. Use ``dpdfnet2_48khz_hr`` when you
   want 48 kHz enhancement output.

Download pre-trained models
---------------------------

Please use the following commands to download DPDFNet ONNX models and a test
wave file:

.. code-block:: bash

   wget https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/baseline.onnx
   wget https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/dpdfnet2.onnx
   wget https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/dpdfnet4.onnx
   wget https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/dpdfnet8.onnx
   wget https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/dpdfnet2_48khz_hr.onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

After downloading, you should have files similar to the following:

.. code-block:: bash

   ls -lh *.onnx inp_16k.wav

Please refer to :doc:`./dpdfnet-python-api` for Python usage and
:doc:`./dpdfnet-c-api` for C API examples.

Demo and project links
----------------------

You can listen to samples and try the online demo at

  - `Project page <https://ceva-ip.github.io/DPDFNet/>`_
  - `Hugging Face demo space <https://huggingface.co/spaces/Ceva-IP/DPDFNetDemo>`_

Citation
--------

.. code-block:: bibtex

   @article{rika2025dpdfnet,
     title = {DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN},
     author = {Rika, Daniel and Sapir, Nino and Gus, Ido},
     year = {2025},
   }
