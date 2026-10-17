offline_speaker_diarization
===========================

Run offline speaker diarization with pyannote segmentation and 3D-Speaker embeddings.

Source file
-----------

`rust-api-examples/examples/offline_speaker_diarization.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/offline_speaker_diarization.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-offline-speaker-diarization.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-offline-speaker-diarization.sh>`_

.. code-block:: bash

  ./run-offline-speaker-diarization.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example offline_speaker_diarization
