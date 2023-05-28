#!/bin/bash
stage=-2
stop_stage=2


pretrained_model_dir=/workspace/icefall/egs/wenetspeech/ASR/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming
model_repo_path=./model_repo_streaming

# modify model specific parameters according to $pretrained_model_dir/exp/onnx_export.log
VOCAB_SIZE=5537

DECODER_CONTEXT_SIZE=2
DECODER_DIM=512

ENCODER_LAYERS=24
ENCODER_DIM=384
CNN_MODULE_KERNEL=31

# for streaming ASR 
ENCODER_LEFT_CONTEXT=32
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
recipe_dir=$icefall_dir/egs/wenetspeech/ASR/pruned_transducer_stateless5

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  if [ -d "$pretrained_model_dir" ]
  then
    echo "skip download pretrained model"
  else
    pushd $icefall_dir/egs/wenetspeech/ASR/
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming
    pushd icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming
    git lfs pull --include "exp/pretrained_epoch_7_avg_1.pt,data/lang_char/Linv.pt"
    popd
    ln -s ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/exp/pretrained_epoch_7_avg_1.pt ./icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming/exp/epoch-999.pt 
    popd
  fi
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "export onnx"
    cp scripts/*onnx*.py ${recipe_dir}/
    cd ${recipe_dir}
    ./export_onnx.py \
        --exp-dir ${pretrained_model_dir}/exp \
        --tokenizer-file $TOKENIZER_FILE \
        --epoch 999 \
        --avg 1 \
        --streaming-model 1\
        --causal-convolution 1 \
        --onnx 1 \
        --left-context $ENCODER_LEFT_CONTEXT \
        --right-context $ENCODER_RIGHT_CONTEXT \
        --fp16
    cd -
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     echo "auto gen config.pbtxt"
     dirs="encoder decoder feature_extractor joiner scorer transducer"

     if [ ! -d $model_repo_path ]; then
        echo "Please cd to $model_repo_path"
        exit 1
     fi

     cp -r $TOKENIZER_FILE $model_repo_path/scorer/
     TOKENIZER_FILE=$model_repo_path/scorer/$(basename $TOKENIZER_FILE)
     for dir in $dirs
     do   
          cp $model_repo_path/$dir/config.pbtxt.template $model_repo_path/$dir/config.pbtxt

          sed -i "s|VOCAB_SIZE|${VOCAB_SIZE}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|DECODER_CONTEXT_SIZE|${DECODER_CONTEXT_SIZE}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|DECODER_DIM|${DECODER_DIM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|ENCODER_LAYERS|${ENCODER_LAYERS}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|ENCODER_DIM|${ENCODER_DIM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|ENCODER_LEFT_CONTEXT|${ENCODER_LEFT_CONTEXT}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|ENCODER_RIGHT_CONTEXT|${ENCODER_RIGHT_CONTEXT}|g" $model_repo_path/$dir/config.pbtxt

          sed -i "s|TOKENIZER_FILE|${TOKENIZER_FILE}|g" $model_repo_path/$dir/config.pbtxt

          sed -i "s|MAX_BATCH|${MAX_BATCH}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|CNN_MODULE_KERNEL_MINUS_ONE|${CNN_MODULE_KERNEL_MINUS_ONE}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|DECODE_WINDOW_SIZE|${DECODE_WINDOW_SIZE}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|DECODE_CHUNK_SIZE|${DECODE_CHUNK_SIZE}|g" $model_repo_path/$dir/config.pbtxt
          
          sed -i "s|FEATURE_EXTRACTOR_INSTANCE_NUM|${FEATURE_EXTRACTOR_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|ENCODER_INSTANCE_NUM|${ENCODER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|JOINER_INSTANCE_NUM|${JOINER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|DECODER_INSTANCE_NUM|${DECODER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|SCORER_INSTANCE_NUM|${SCORER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt

     done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    cp $pretrained_model_dir/exp/encoder_fp16.onnx $model_repo_path/encoder/1/encoder.onnx
    cp $pretrained_model_dir/exp/decoder_fp16.onnx $model_repo_path/decoder/1/decoder.onnx
    cp $pretrained_model_dir/exp/joiner_fp16.onnx $model_repo_path/joiner/1/joiner.onnx
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    tritonserver --model-repository=$model_repo_path --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000 --http-port 10086
fi
