DPDFNet Python API
==================

This page describes how to use the Python API for DPDFNet with `sherpa-onnx`_.

See :ref:`install_sherpa_onnx_python` for how to install the
Python package of `sherpa-onnx`_.

The following is a quick way to do that::

  pip install sherpa-onnx soundfile

Offline speech enhancement
--------------------------

Download a DPDFNet model and a test wave file:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

All of the 8, 16, and 48 kHz models in :doc:`./dpdfnet` can be used with
this API.

The following example uses
``OfflineSpeechDenoiserDpdfNetModelConfig`` and
``OfflineSpeechDenoiser``:

.. code-block:: python

   import numpy as np
   import sherpa_onnx
   import soundfile as sf


   def load_audio(filename: str):
       samples, sample_rate = sf.read(
           filename,
           always_2d=True,
           dtype="float32",
       )
       samples = np.ascontiguousarray(samples[:, 0])
       return samples, sample_rate


   config = sherpa_onnx.OfflineSpeechDenoiserConfig(
       model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
           dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(
               model="./dpdfnet2.onnx",
               attenuation_limit_db=12.0,
           ),
           num_threads=1,
           debug=False,
           provider="cpu",
       )
   )

   assert config.validate(), config

   denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
   samples, sample_rate = load_audio("./inp_16k.wav")
   denoised = denoiser.run(samples, sample_rate)

   sf.write("enhanced.wav", denoised.samples, denoised.sample_rate)
   print(f"Saved to enhanced.wav at {denoised.sample_rate} Hz")

``attenuation_limit_db`` is optional and defaults to ``0`` (disabled). A
positive value limits offline suppression by blending aligned noisy spectra
back into the enhanced spectra. Finite values must be in ``[0, 100]``;
infinity also disables the limit. See :ref:`dpdfnet-offline-attenuation-limit`
for the exact behavior.

You can also run the upstream example directly:

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa-onnx
   cd sherpa-onnx
   python3 ./python-api-examples/offline-speech-enhancement-dpdfnet.py

The example script is available at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speech-enhancement-dpdfnet.py>`_

Online streaming speech enhancement
-----------------------------------

The Python API also provides ``OnlineSpeechDenoiser``. The following example
feeds one model frame shift at a time and calls ``flush()`` to retrieve the
tail and reset the stream:

.. code-block:: python

   config = sherpa_onnx.OnlineSpeechDenoiserConfig(
       model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
           dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(
               model="./dpdfnet2.onnx",
           ),
           num_threads=1,
           debug=False,
           provider="cpu",
       )
   )

   assert config.validate(), config

   denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)
   samples, sample_rate = load_audio("./inp_16k.wav")
   output = []

   for start in range(0, len(samples), denoiser.frame_shift_in_samples):
       chunk = samples[start : start + denoiser.frame_shift_in_samples]
       denoised = denoiser.run(chunk, sample_rate)
       output.append(np.asarray(denoised.samples, dtype=np.float32))

   output.append(
       np.asarray(denoiser.flush().samples, dtype=np.float32)
   )
   enhanced = np.concatenate(output)
   sf.write("enhanced-streaming.wav", enhanced, denoiser.sample_rate)

The streaming denoiser supports every model listed in :doc:`./dpdfnet` and
resamples input to the model's native sample rate when necessary. The input
sample rate must stay fixed until ``flush()`` or ``reset()``. The offline
``attenuation_limit_db`` setting is not applied in streaming mode.

You can also run the upstream streaming example directly:

.. code-block:: bash

   python3 ./python-api-examples/online-speech-enhancement-dpdfnet.py

The streaming example source is available at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-speech-enhancement-dpdfnet.py>`_

Hints
-----

You can try DPDFNet in your browser at
`Ceva-IP/DPDFNetDemo <https://huggingface.co/spaces/Ceva-IP/DPDFNetDemo>`_
and download the model files from the
`speech-enhancement-models release <https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models>`_.
