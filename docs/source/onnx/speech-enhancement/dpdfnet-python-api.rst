DPDFNet Python API
==================

This page describes how to use the Python API for DPDFNet with `sherpa-onnx`_.

Please refer to :ref:`install_sherpa_onnx_python` for how to install the
Python package of `sherpa-onnx`_.

The following is a quick way to do that::

  pip install sherpa-onnx soundfile

Offline speech enhancement
--------------------------

Download a DPDFNet model and a test wave file:

.. code-block:: bash

   wget https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/dpdfnet2.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

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

You can also run the upstream example directly:

.. code-block:: bash

   git clone https://github.com/k2-fsa/sherpa-onnx
   cd sherpa-onnx
   python3 ./python-api-examples/offline-speech-enhancement-dpdfnet.py

The example script is available at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speech-enhancement-dpdfnet.py>`_

Streaming status
----------------

The current ``dpdfnet`` branch of `sherpa-onnx`_ contains the offline Python
bindings and offline example shown above.

.. note::

   The online streaming DPDFNet denoiser is available in the core runtime and
   C API on the current branch, but a corresponding Python binding/example is
   not present in this branch yet. Please refer to :doc:`./dpdfnet-c-api` for
   streaming usage.

Hints
-----

You can try DPDFNet in your browser at
`Ceva-IP/DPDFNetDemo <https://huggingface.co/spaces/Ceva-IP/DPDFNetDemo>`_
and download the official model files from
`Ceva-IP/DPDFNet <https://huggingface.co/Ceva-IP/DPDFNet>`_.
