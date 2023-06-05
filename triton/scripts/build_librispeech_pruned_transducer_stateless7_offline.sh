#!/bin/bash
stage=0
stop_stage=2

# change to your own model directory
pretrained_model_dir=/mnt/samsung-t7/wend/github/icefall/egs/librispeech/ASR/pruned_transducer_stateless7/exp/
model_repo_path=./zipformer/model_repo_offline

# modify model specific parameters according to $pretrained_model_dir/exp/onnx_export.log
VOCAB_SIZE=500

DECODER_CONTEXT_SIZE=2
DECODER_DIM=512

ENCODER_LAYERS=12
ENCODER_DIM=512
CNN_MODULE_KERNEL=31

if [ -d "$pretrained_model_dir/data/lang_char" ]
then
    echo "pretrained model using char"
    TOKENIZER_FILE_OR_DIR=$pretrained_model_dir/data/lang_char
else
    echo "pretrained model using bpe"
    TOKENIZER_FILE_OR_DIR=$pretrained_model_dir/data/lang_bpe_500/bpe.model
fi

MAX_BATCH=512
# model instance num
FEATURE_EXTRACTOR_INSTANCE_NUM=2
ENCODER_INSTANCE_NUM=2
JOINER_INSTANCE_NUM=1
DECODER_INSTANCE_NUM=1
SCORER_INSTANCE_NUM=2


icefall_dir=/mnt/samsung-t7/wend/asr/icefall
export PYTHONPATH=$PYTHONPATH:$icefall_dir
recipe_dir=$icefall_dir/egs/librispeech/ASR/pruned_transducer_stateless7

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  if [ -d "$pretrained_model_dir" ]
  then
    echo "skip downloading pretrained model"
  else
    pushd $icefall_dir/egs/librispeech/ASR/
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
    pushd icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
    git lfs pull --include "exp/pretrained-epoch-30-avg-9.pt"
    git lfs pull --include "data/lang_bpe_500/bpe.model"
    ln -rs icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/pretrained-epoch-30-avg-9.pt icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/epoch-999.pt
    popd
  fi
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "export onnx"
    cd ${recipe_dir}
    ./export_onnx.py \
        --exp-dir ${pretrained_model_dir}/exp \
        --tokenizer-file $TOKENIZER_FILE_OR_DIR \
        --epoch 999 \
        --avg 1 \
        --use-averaged-model 0
    cd -
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     echo "auto gen config.pbtxt"
     dirs="encoder decoder feature_extractor joiner scorer transducer"

     if [ ! -d $model_repo_path ]; then
        echo "Please cd to $model_repo_path"
        exit 1
     fi

     cp -r $TOKENIZER_FILE_OR_DIR $model_repo_path/scorer/
     TOKENIZER_FILE=$model_repo_path/scorer/$(basename $TOKENIZER_FILE_OR_DIR)
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

          sed -i "s|FEATURE_EXTRACTOR_INSTANCE_NUM|${FEATURE_EXTRACTOR_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|ENCODER_INSTANCE_NUM|${ENCODER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|JOINER_INSTANCE_NUM|${JOINER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|DECODER_INSTANCE_NUM|${DECODER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|SCORER_INSTANCE_NUM|${SCORER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt

     done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    cp $pretrained_model_dir/exp/encoder-epoch-999-avg-1.onnx $model_repo_path/encoder/1/encoder.onnx
    cp $pretrained_model_dir/exp/decoder-epoch-999-avg-1.onnx $model_repo_path/decoder/1/decoder.onnx
    cp $pretrained_model_dir/exp/joiner-epoch-999-avg-1.onnx $model_repo_path/joiner/1/joiner.onnx
    cp $TOKENIZER_FILE /workspace/
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    tritonserver --model-repository=$model_repo_path --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000 --http-port 10086
fi

