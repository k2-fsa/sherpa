DPDFNet
=======

`DPDFNet <https://github.com/ceva-ip/DPDFNet>`_ is a family of causal,
single-channel speech enhancement models for real-time noise suppression.
It extends DeepFilterNet2 with Dual-Path RNN (DPRNN) blocks in the encoder
for stronger long-range temporal and cross-band modeling while staying
streaming-friendly. The paper is available on
`arXiv <https://arxiv.org/abs/2512.16420>`_. The source project is hosted at
`GitHub <https://github.com/ceva-ip/DPDFNet>`_ and the pre-trained ONNX models
used by `sherpa-onnx`_ are published in the
`speech-enhancement-models release <https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models>`_.
They are also available from the
`official Hugging Face repository <https://huggingface.co/Ceva-IP/DPDFNet>`_.

In `sherpa-onnx`_, DPDFNet supports offline speech enhancement and online
streaming speech enhancement. Both modes support the official 8, 16, and
48 kHz ONNX exports listed below. The input is resampled when necessary, and
the enhanced audio uses the model's native sample rate.

.. note::

   The full set of DPDFNet ONNX models and sample wave files such as
   ``inp_16k.wav`` and ``speech_with_noise.wav`` are available from the
   ``speech-enhancement-models`` GitHub release.

Model variants
--------------

8 kHz models
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Model
     - Params (M)
     - MACs (G)
     - Intended use
   * - ``dpdfnet2_8khz``
     - 2.51
     - 1.29
     - Low-bandwidth real-time enhancement
   * - ``dpdfnet8_8khz``
     - 3.56
     - 3.99
     - Best 8 kHz enhancement quality

16 kHz models
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Model
     - Params (M)
     - MACs (G)
     - Intended use
   * - ``dpdfnet_baseline``
     - 2.31
     - 0.36
     - Fastest / lowest resource usage
   * - ``dpdfnet2``
     - 2.49
     - 1.35
     - Real-time / embedded devices
   * - ``dpdfnet4``
     - 2.84
     - 2.36
     - Balanced performance
   * - ``dpdfnet8``
     - 3.54
     - 4.37
     - Best enhancement quality

48 kHz models
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Model
     - Params (M)
     - MACs (G)
     - Intended use
   * - ``dpdfnet2_48khz_hr``
     - 2.58
     - 2.42
     - Balanced high-resolution enhancement
   * - ``dpdfnet8_48khz_hr``
     - 3.63
     - 7.17
     - Best 48 kHz enhancement quality

.. hint::

   Use ``dpdfnet2_8khz`` or ``dpdfnet8_8khz`` for 8 kHz audio;
   ``dpdfnet_baseline``, ``dpdfnet2``, ``dpdfnet4``, or ``dpdfnet8`` for
   16 kHz downstream ASR or speech recognition; and
   ``dpdfnet2_48khz_hr`` or ``dpdfnet8_48khz_hr`` for 48 kHz enhancement.

Download pre-trained models
---------------------------

Please use the following commands to download DPDFNet ONNX models and a test
wave file:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet4.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet8.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2_48khz_hr.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2_8khz.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet8_8khz.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet8_48khz_hr.onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

After downloading, you should have files similar to the following:

.. code-block:: bash

   ls -lh *.onnx inp_16k.wav

.. _dpdfnet-offline-attenuation-limit:

Offline attenuation limit
-------------------------

DPDFNet's offline configuration has an optional attenuation limit. It reduces
over-suppression by mixing an aligned copy of the noisy spectrum back into the
enhanced spectrum. For a positive limit ``L`` in dB, the noisy-signal weight is
``10^(-L / 20)`` and the enhanced-signal weight is the remainder. For example,
``12`` dB uses a noisy-signal weight of about ``0.251``.

The default value, ``0``, disables the limit and preserves the model's full
suppression. Finite values in ``[0, 100]`` are valid; ``0`` and infinity
disable the limit. Among positive values, a larger value permits stronger
suppression because it mixes in less of the noisy spectrum.

The command-line option is
``--speech-denoiser-dpdfnet-attenuation-limit-db``. The corresponding binding
names are:

.. list-table::
   :header-rows: 1

   * - API
     - Offline DPDFNet configuration field
   * - C, C++, Python, Rust
     - ``attenuation_limit_db``
   * - Dart / Flutter, JavaScript / WebAssembly / HarmonyOS, Kotlin, Swift
     - ``attenuationLimitDb``
   * - .NET, Go, Pascal
     - ``AttenuationLimitDb``
   * - Java
     - ``setAttenuationLimitDb(float)`` on the builder

.. important::

   The attenuation limit is implemented only by the offline DPDFNet denoiser.
   It is not applied during online streaming enhancement; leave it at its
   default value in an online configuration.

Command-line examples
---------------------

The offline command accepts the attenuation limit and every model variant:

.. code-block:: bash

   ./bin/sherpa-onnx-offline-denoiser \
     --speech-denoiser-dpdfnet-model=dpdfnet8_8khz.onnx \
     --speech-denoiser-dpdfnet-attenuation-limit-db=12 \
     --input-wav=input.wav \
     --output-wav=enhanced-8k.wav

The online command supports the same model variants, without the offline-only
attenuation behavior:

.. code-block:: bash

   ./bin/sherpa-onnx-online-denoiser \
     --speech-denoiser-dpdfnet-model=dpdfnet8_48khz_hr.onnx \
     --chunk-duration-ms=10 \
     --input-wav=input.wav \
     --output-wav=enhanced-48k.wav

See :doc:`./dpdfnet-python-api` for Python usage and
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
