#!/bin/bash

set -e # exit on error

stage=-2
stop_stage=2


pretrained_model_dir=/workspace/icefall/egs/librispeech/ASR/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
model_repo_path=$(pwd)/model_repo_streaming_zipformer_test
#conformer_streaming_model_repo_path=$(pwd)/model_repo_streaming

# modify model specific parameters according to $pretrained_model_dir/exp/onnx_export.log
VOCAB_SIZE=500

DECODER_CONTEXT_SIZE=2
DECODER_DIM=512

#ENCODER_LAYERS=24

ENCODER_LAYERS=15
ENCODER_LAYERS_2X=$((2*$ENCODER_LAYERS))
ENCODER_LAYERS_3X=$((3*$ENCODER_LAYERS))


ENCODER_DIM=384
ENCODER_DIM_HALF=$(($ENCODER_DIM/2))
CNN_MODULE_KERNEL=31

# for streaming ASR 
ENCODER_LEFT_CONTEXT=64
ENCODER_RIGHT_CONTEXT=2

if [ -d "$pretrained_model_dir/data/lang_char" ] 
then
    echo "pretrained model using char"
    TOKENIZER_FILE=$pretrained_model_dir/data/lang_char
else
    echo "pretrained model using bpe"
    TOKENIZER_FILE=$pretrained_model_dir/data/lang_bpe_500/bpe.model
fi

MAX_BATCH=512
# for streaming ASR 
CNN_MODULE_KERNEL_MINUS_ONE=$(($CNN_MODULE_KERNEL - 1))
DECODE_CHUNK_SIZE=16
# decode_window_size = (decode_chunk_size + 2 + decode_right_context) * subsampling_factor + 3 
DECODE_WINDOW_SIZE=$((($DECODE_CHUNK_SIZE+2+$ENCODER_RIGHT_CONTEXT)*4+3))
# model instance num
FEATURE_EXTRACTOR_INSTANCE_NUM=2
ENCODER_INSTANCE_NUM=2
JOINER_INSTANCE_NUM=1
DECODER_INSTANCE_NUM=1
SCORER_INSTANCE_NUM=2

icefall_dir=/workspace/icefall
export PYTHONPATH=$PYTHONPATH:$icefall_dir
recipe_dir=$icefall_dir/egs/librispeech/ASR/pruned_transducer_stateless7_streaming

FP_32=true

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  if [ -d "$pretrained_model_dir" ]
  then
    echo "skip download pretrained model"
  else

    cd $icefall_dir/egs/librispeech/ASR/
    MODEL_LOCATION=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
    GIT_LFS_SKIP_SMUDGE=1 git clone $MODEL_LOCATION
    cd icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
    git lfs pull --include "exp/epoch-30.pt"
    git lfs pull --include "data/lang_bpe_500/bpe.model"

  fi
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "export onnx"

    cd ${recipe_dir}

    if [ "$FP_32" = false ] ; then

    echo "Using FP16"

    ./export.py \
        --exp-dir ${pretrained_model_dir}/exp \
        --bpe-model $TOKENIZER_FILE \
        --use-averaged-model False \
        --epoch 30 \
        --avg 1 \
        --fp16 \
        --onnx-triton 1 \
        --onnx 1

    sed -i "s|TYPE_FP32|TYPE_FP16|g" "${model_repo_path}"/*/config.pbtxt.template

    else 

    echo "Using FP32"

    ./export.py \
        --exp-dir ${pretrained_model_dir}/exp \
        --bpe-model $TOKENIZER_FILE \
        --use-averaged-model False \
        --epoch 30 \
        --avg 1 \
        --onnx-triton 1 \
        --onnx 1

    sed -i "s|TYPE_FP16|TYPE_FP32|g" "${model_repo_path}"/*/config.pbtxt.template

    fi

    cd -
fi



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     echo "auto gen config.pbtxt"
     dirs="decoder encoder feature_extractor joiner scorer transducer"

     if [ ! -d $model_repo_path ]; then
        echo "Please cd to $model_repo_path"
        exit 1
     fi

     cp -r $TOKENIZER_FILE $model_repo_path/scorer/


     TOKENIZER_FILE=$model_repo_path/scorer/$(basename $TOKENIZER_FILE)
     for dir in $dirs
     do   
          cp $model_repo_path/$dir/config.pbtxt.template $model_repo_path/$dir/config.pbtxt
     done

      sed -i "s|ENCODER_LAYERS_2X|${ENCODER_LAYERS_2X}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_LAYERS_3X|${ENCODER_LAYERS_3X}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_DIM_HALF|${ENCODER_DIM_HALF}|g" $model_repo_path/*/config.pbtxt


      sed -i "s|VOCAB_SIZE|${VOCAB_SIZE}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|DECODER_CONTEXT_SIZE|${DECODER_CONTEXT_SIZE}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|DECODER_DIM|${DECODER_DIM}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_LAYERS|${ENCODER_LAYERS}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_DIM|${ENCODER_DIM}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_LEFT_CONTEXT|${ENCODER_LEFT_CONTEXT}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_RIGHT_CONTEXT|${ENCODER_RIGHT_CONTEXT}|g" $model_repo_path/*/config.pbtxt

      sed -i "s|TOKENIZER_FILE|${TOKENIZER_FILE}|g" $model_repo_path/*/config.pbtxt

      sed -i "s|MAX_BATCH|${MAX_BATCH}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|CNN_MODULE_KERNEL_MINUS_ONE|${CNN_MODULE_KERNEL_MINUS_ONE}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|DECODE_WINDOW_SIZE|${DECODE_WINDOW_SIZE}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|DECODE_CHUNK_SIZE|${DECODE_CHUNK_SIZE}|g" $model_repo_path/*/config.pbtxt
      
      sed -i "s|FEATURE_EXTRACTOR_INSTANCE_NUM|${FEATURE_EXTRACTOR_INSTANCE_NUM}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_INSTANCE_NUM|${ENCODER_INSTANCE_NUM}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|JOINER_INSTANCE_NUM|${JOINER_INSTANCE_NUM}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|DECODER_INSTANCE_NUM|${DECODER_INSTANCE_NUM}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|SCORER_INSTANCE_NUM|${SCORER_INSTANCE_NUM}|g" $model_repo_path/*/config.pbtxt

      sed -i "s|ENCODER_LAYERS_2X|${ENCODER_LAYERS_2X}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_LAYERS_3X|${ENCODER_LAYERS_3X}|g" $model_repo_path/*/config.pbtxt
      sed -i "s|ENCODER_DIM_HALF|${ENCODER_DIM_HALF}|g" $model_repo_path/*/config.pbtxt


fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

  if [ "$FP_32" = true ] ; then

    cp -f $pretrained_model_dir/exp/encoder.onnx $model_repo_path/encoder/1/encoder.onnx
    cp -f $pretrained_model_dir/exp/decoder.onnx $model_repo_path/decoder/1/decoder.onnx
    cp -f $pretrained_model_dir/exp/joiner.onnx $model_repo_path/joiner/1/joiner.onnx

  else 

    cp -f $pretrained_model_dir/exp/encoder_fp16.onnx $model_repo_path/encoder/1/encoder.onnx
    cp -f $pretrained_model_dir/exp/decoder_fp16.onnx $model_repo_path/decoder/1/decoder.onnx
    cp -f $pretrained_model_dir/exp/joiner_fp16.onnx $model_repo_path/joiner/1/joiner.onnx


  fi



fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    tritonserver --model-repository=$model_repo_path --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000 --http-port 10086
fi
