Triton-server
=============
This page gives serveral examples to deploy streaming and offline ASR pretrained models with Triton server.

Deploy streaming ASR models with Onnx 
-------------------------------------

First, we need to export pretrained models with Onnx.

.. code-block:: bash

   export SHERPA_SRC=./sherpa
   export ICEFALL_SRC=/workspace/icefall
   # copy essentials
   cp $SHERPA_SRC/triton/scripts/*onnx*.py $ICEFALL_DIR/egs/wenetspeech/ASR/pruned_stateless_transducer5/
   cd $ICEFALL_SRC/egs/wenetspeech/ASR/
   # download pretrained models
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming
   cd ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming
   git lfs pull --include "exp/pretrained_epoch_7_avg_1.pt"
   cd -
   # export to onnx fp16
   ln -s ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/exp/pretrained_epoch_7_avg_1.pt ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/exp/epoch-999.pt 
   ./pruned_transducer_stateless5/export_onnx.py \
      --exp-dir ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/exp \
      --tokenizer-file ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/data/lang_char \
      --epoch 999 \
      --avg 1 \
      --streaming-model 1 \
      --causal-convolution 1 \
      --onnx 1 \
      --left-context 64 \
      --right-context 4 \
      --fp16

.. note::

   For Chinese models, ``--tokenizer-file`` points to ``<pretrained_dir>/data/lang_char``. While for English models, it points to ``<pretrained_dir>/data/lang_bpe_500/bpe.model`` file.

Then, in the docker container, you could start the service with:

.. code-block:: bash

   cd sherpa/triton/
   bash scripts/start_streaming_server.sh



Deploy offline ASR models with torchscript 
------------------------------------------
.. caution::
   Currently, we only support FP32 offline ASR inference for torchscript backend. Streaming ASR and FP16 inference are not supported.

First, we need to export pretrained models using jit.

.. code-block:: bash

   export SHERPA_SRC=./sherpa
   export ICEFALL_SRC=/workspace/icefall
   # Download pretrained models
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-04-29 $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3/
   cd icefall-asr-librispeech-pruned-transducer-stateless3-2022-04-29
   git lfs pull --include "exp/pretrained-epoch-25-avg-7.pt"
   # export them to three jit models: encoder_jit.pt, decoder_jit.pt, joiner_jit.pt
   cp $SHERPA_SRC/triton/scripts/conformer_triton.py $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3/
   cp $SHERPA_SRC/triton/scripts/export_jit.py $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3/
   cd $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3
   python3 export_jit.py \
           --pretrained-model $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3/icefall-asr-librispeech-pruned-transducer-stateless3-2022-04-29 \
           --output-dir <jit_model_dir> --bpe-model <bpe_model_path>
   # copy bpe file to <jit_model_dir>, later we would mount <jit_model_dir> to the triton docker container
   cp <bpe_model_path> <jit_model_dir>

.. note::

   If you export models outside the docker container, you could mount the exported ``<jit_model_dir>`` with 
   ``-v <host_dir>:<container_dir>`` when lauching the container.

Then, in the docker container, you could start the service with:

.. code-block:: bash
   
   cd sherpa/triton/
   bash scripts/start_offline_server_jit.sh
