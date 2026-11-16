Pre-trained models
==================

This page lists pre-trained models for speaker segmentation.

Models for speaker embedding extraction can be found at

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models>`_

Colab notebook
--------------


We provide a colab notebook
|speaker diarization with sherpa-onnx colab notebook|
for you to try this section step by step.

.. |speaker diarization with sherpa-onnx colab notebook| image:: https://github.com/k2-fsa/sherpa/releases/download/doc/colab-badge.jpg
   :target: https://github.com/k2-fsa/colab/blob/master/sherpa-onnx/sherpa_onnx_speaker_diarization.ipynb

sherpa-onnx-pyannote-segmentation-3-0
-------------------------------------

This model is converted from `<https://huggingface.co/pyannote/segmentation-3.0>`_.
You can find the conversion script at `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/pyannote/segmentation>`_.

The sections below show how to use it together with
a speaker embedding extraction model for speaker diarization.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following code to download the model:

.. code-block:: bash

   cd /path/to/sherpa-onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
   tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
   rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

   ls -lh sherpa-onnx-pyannote-segmentation-3-0/{*.onnx,LICENSE,README.md}

You should see the following output::

  -rw-r--r--  1 fangjun  staff   1.0K Oct  8 20:54 sherpa-onnx-pyannote-segmentation-3-0/LICENSE
  -rw-r--r--  1 fangjun  staff   115B Oct  8 20:54 sherpa-onnx-pyannote-segmentation-3-0/README.md
  -rw-r--r--  1 fangjun  staff   1.5M Oct  8 20:54 sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx
  -rw-r--r--  1 fangjun  staff   5.7M Oct  8 20:54 sherpa-onnx-pyannote-segmentation-3-0/model.onnx

Usage for speaker diarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's download a test wave file. The model expects wave files of 16kHz, 16-bit and a single channel.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Next, let's download a model for extracting speaker embeddings. You can find lots of models from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models>`_. We
download two models in this example::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx

Now let's run it.

3D-Speaker + model.onnx
:::::::::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --clustering.num-clusters=4 \
     --segmentation.pyannote-model=./sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
     --embedding.model=./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
     ./0-four-speakers-zh.wav

   # Note: Since we know there are 4 speakers in ./0-four-speakers-zh.wav file, we
   # provide the argument --clustering.num-clusters=4.
   # If you don't have such information, please use the argument --clustering.cluster-threshold.
   # A larger threshold results in fewer speakers.
   # A smaller threshold results in more speakers.
   #
   # Hint: You can use --clustering.cluster-threshold=0.9 for this specific wave file.

The output is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/pyannote-segmentation-3-0-3dspeaker.txt

3D-Speaker + model.int8.onnx
:::::::::::::::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --clustering.num-clusters=4 \
     --segmentation.pyannote-model=./sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx \
     --embedding.model=./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
     ./0-four-speakers-zh.wav

The output is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/pyannote-segmentation-3-0-3dspeaker.int8.txt


NeMo + model.onnx
:::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --clustering.num-clusters=4 \
     --segmentation.pyannote-model=./sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
     --embedding.model=./nemo_en_titanet_small.onnx \
     ./0-four-speakers-zh.wav

The output is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/pyannote-segmentation-3-0-nemo.txt

NeMo + model.int8.onnx
::::::::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --clustering.num-clusters=4 \
     --segmentation.pyannote-model=./sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx \
     --embedding.model=./nemo_en_titanet_small.onnx \
     ./0-four-speakers-zh.wav

The output is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/pyannote-segmentation-3-0-nemo.int8.txt

sherpa-onnx-reverb-diarization-v1
---------------------------------

This model is converted from `<https://huggingface.co/Revai/reverb-diarization-v1>`_.
You can find the conversion script at `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/pyannote/segmentation>`_.

.. caution::

   It is accessible under a ``non-commercial`` license.
   You can find its license at `<https://huggingface.co/Revai/reverb-diarization-v1/blob/main/LICENSE>`_.

The sections below show how to use it together with
a speaker embedding extraction model for speaker diarization.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following code to download the model:

.. code-block:: bash

   cd /path/to/sherpa-onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2
   tar xvf sherpa-onnx-reverb-diarization-v1.tar.bz2
   rm sherpa-onnx-reverb-diarization-v1.tar.bz2

   ls -lh sherpa-onnx-reverb-diarization-v1/{*.onnx,LICENSE,README.md}

You should see the following output::

  -rw-r--r--  1 fangjun  staff    11K Oct 17 10:49 sherpa-onnx-reverb-diarization-v1/LICENSE
  -rw-r--r--  1 fangjun  staff   320B Oct 17 10:49 sherpa-onnx-reverb-diarization-v1/README.md
  -rw-r--r--  1 fangjun  staff   2.3M Oct 17 10:49 sherpa-onnx-reverb-diarization-v1/model.int8.onnx
  -rw-r--r--  1 fangjun  staff   9.1M Oct 17 10:49 sherpa-onnx-reverb-diarization-v1/model.onnx

Usage for speaker diarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's download a test wave file. The model expects wave files of 16kHz, 16-bit and a single channel.

.. code-block:: bash

   cd /path/to/sherpa-onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Next, let's download a model for extracting speaker embeddings. You can find lots of models from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models>`_. We
download two models in this example::

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx

Now let's run it.

3D-Speaker + model.onnx
:::::::::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --clustering.num-clusters=4 \
     --segmentation.pyannote-model=./sherpa-onnx-reverb-diarization-v1/model.onnx \
     --embedding.model=./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
     ./0-four-speakers-zh.wav

   # Note: Since we know there are 4 speakers in ./0-four-speakers-zh.wav file, we
   # provide the argument --clustering.num-clusters=4.
   # If you don't have such information, please use the argument --clustering.cluster-threshold.
   # A larger threshold results in fewer speakers.
   # A smaller threshold results in more speakers.
   #
   # Hint: You can use --clustering.cluster-threshold=0.9 for this specific wave file.

The output is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/revai-segmentation-3-0-3dspeaker.txt

3D-Speaker + model.int8.onnx
::::::::::::::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --clustering.num-clusters=4 \
     --segmentation.pyannote-model=./sherpa-onnx-reverb-diarization-v1/model.int8.onnx \
     --embedding.model=./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
     ./0-four-speakers-zh.wav

The output is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/revai-segmentation-3-0-3dspeaker.int8.txt

NeMo + model.onnx
:::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --clustering.num-clusters=4 \
     --segmentation.pyannote-model=./sherpa-onnx-reverb-diarization-v1/model.onnx \
     --embedding.model=./nemo_en_titanet_small.onnx \
     ./0-four-speakers-zh.wav

The output is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/revai-segmentation-3-0-nemo.txt

NeMo + model.int8.onnx
::::::::::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --clustering.num-clusters=4 \
     --segmentation.pyannote-model=./sherpa-onnx-reverb-diarization-v1/model.int8.onnx \
     --embedding.model=./nemo_en_titanet_small.onnx \
     ./0-four-speakers-zh.wav

The output is given below:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/revai-segmentation-3-0-nemo.int8.txt

.. _sherpa-onnx-nemotron-3-diarization:

sherpa-onnx-nemotron-3-diarization
----------------------------------

This model is converted from `<https://huggingface.co/nvidia/Nemotron-3-Diarization>`_.
The conversion script is at `<https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/nemotron-3-diarization>`_.

It is a `Sortformer <https://arxiv.org/abs/2409.06656>`_ model. One model does all of the
speaker diarization. This is different from the models above:

  - You do not need a speaker embedding model.
  - You do not need clustering. Do not set ``--clustering.num-clusters``
    or ``--clustering.cluster-threshold``.
  - The model finds 8 speakers or fewer. It gives speaker numbers in the
    order in which the speakers start to talk.

When to use this model:

  - Use it if your recordings have 8 speakers or fewer, and you do not know
    the number of speakers.
  - Use a pyannote model with an embedding model if your recordings
    can have more than 8 speakers.

.. caution::

   The license of this model is ``OpenMDW-1.1``.
   The model archive has the full license in the file ``LICENSE``.

Download the model
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd /path/to/sherpa-onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-nemotron-3-diarization.tar.bz2
   tar xvf sherpa-onnx-nemotron-3-diarization.tar.bz2
   rm sherpa-onnx-nemotron-3-diarization.tar.bz2

The directory has these files:

  - ``model.onnx``: the float32 model (380 MB)
  - ``model.int8.onnx``: the int8 model (100 MB). It is smaller and faster.
    Its results can be a little different from ``model.onnx``.
  - ``test_wavs/0-four-speakers-zh.wav``: a test wave file
  - ``LICENSE``: the license of the model
  - ``MODEL_CARD.md``: the model card from NVIDIA
  - ``README.md``, ``NOTICE``, ``*.py``, ``requirements.txt``, ``run.sh``:
    the notes and scripts for the conversion

Usage for speaker diarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model expects wave files with a sample rate of 16 kHz, 16-bit samples and one channel.

Download a test wave file:

.. code-block:: bash

   cd /path/to/sherpa-onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Use ``--segmentation.sortformer-model``. Do not use ``--segmentation.pyannote-model``
or ``--embedding.model``.

model.onnx
::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --segmentation.sortformer-model=./sherpa-onnx-nemotron-3-diarization/model.onnx \
     ./0-four-speakers-zh.wav

.. hint::

   ``--segmentation.sortformer-threshold`` sets the speech activity threshold for each
   speaker. The value must be more than 0 and less than 1. The default is 0.5.
   If you increase it, the model finds less speech.

The output is:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/nemotron-3-diarization.txt

model.int8.onnx
:::::::::::::::

.. code-block:: bash

   cd /path/to/sherpa-onnx

   ./build/bin/sherpa-onnx-offline-speaker-diarization \
     --segmentation.sortformer-model=./sherpa-onnx-nemotron-3-diarization/model.int8.onnx \
     ./0-four-speakers-zh.wav

The output is:

.. container:: toggle

    .. container:: header

      Click ▶ to see the output.

    .. literalinclude:: ./code/nemotron-3-diarization.int8.txt

Python
::::::

.. code-block:: python

   import sherpa_onnx
   import soundfile as sf

   config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
       segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
           sortformer=sherpa_onnx.OfflineSpeakerSegmentationSortformerModelConfig(
               model="./sherpa-onnx-nemotron-3-diarization/model.int8.onnx",
           ),
       ),
   )
   sd = sherpa_onnx.OfflineSpeakerDiarization(config)

   # The file must have a sample rate of 16 kHz and one channel.
   audio, sample_rate = sf.read("./0-four-speakers-zh.wav", dtype="float32")
   assert sample_rate == sd.sample_rate

   for r in sd.process(audio).sort_by_start_time():
       print(f"{r.start:.3f} -- {r.end:.3f} speaker_{r.speaker:02}")

The full example is at
`<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization-sortformer.py>`_.
