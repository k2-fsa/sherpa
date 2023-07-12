#!/bin/bash
stage=-2
stop_stage=2

# whether to use convert ONNX model to FP16
fp16=true

pretrained_model_dir=/workspace/icefall/egs/librispeech/ASR/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
model_repo_path=./model_repo_offline

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
    TOKENIZER_FILE=$pretrained_model_dir/data/lang_char
else
    echo "pretrained model using bpe"
    TOKENIZER_FILE=$pretrained_model_dir/data/lang_bpe_500/bpe.model
fi

MAX_BATCH=512
# model instance num
FEATURE_EXTRACTOR_INSTANCE_NUM=2
ENCODER_INSTANCE_NUM=2
JOINER_INSTANCE_NUM=1
DECODER_INSTANCE_NUM=1
SCORER_INSTANCE_NUM=2


icefall_dir=/workspace/icefall
export PYTHONPATH=$PYTHONPATH:$icefall_dir
recipe_dir=$icefall_dir/egs/librispeech/ASR/pruned_transducer_stateless3

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  if [ -d "$pretrained_model_dir" ]
  then
    echo "skip download pretrained model"
  else
    pushd $icefall_dir/egs/librispeech/ASR/
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
    pushd icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
    git lfs pull --include "exp/pretrained-iter-1224000-avg-14.pt"
    git lfs pull --include "data/lang_bpe_500/bpe.model"
    popd
    ln -s ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/pretrained-iter-1224000-avg-14.pt ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/epoch-9999.pt
    popd
  fi
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "export onnx"
    cd $recipe_dir
    ./export-onnx.py \
        --bpe-model $TOKENIZER_FILE \
        --epoch 9999 \
        --avg 1 \
        --exp-dir $pretrained_model_dir/exp/
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
          
          sed -i "s|FEATURE_EXTRACTOR_INSTANCE_NUM|${FEATURE_EXTRACTOR_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|ENCODER_INSTANCE_NUM|${ENCODER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|JOINER_INSTANCE_NUM|${JOINER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|DECODER_INSTANCE_NUM|${DECODER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt
          sed -i "s|SCORER_INSTANCE_NUM|${SCORER_INSTANCE_NUM}|g" $model_repo_path/$dir/config.pbtxt

     done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ $fp16 == true ]; then
	echo "Convert to FP16..."
        polygraphy convert --fp-to-fp16 -o $pretrained_model_dir/exp/encoder_fp16.onnx $pretrained_model_dir/exp/encoder-epoch-9999-avg-1.onnx
        polygraphy convert --fp-to-fp16 -o $pretrained_model_dir/exp/decoder_fp16.onnx $pretrained_model_dir/exp/decoder-epoch-9999-avg-1.onnx
        polygraphy convert --fp-to-fp16 -o $pretrained_model_dir/exp/joiner_fp16.onnx $pretrained_model_dir/exp/joiner-epoch-9999-avg-1.onnx
	
	cp $pretrained_model_dir/exp/encoder_fp16.onnx $model_repo_path/encoder/1/encoder.onnx
	cp $pretrained_model_dir/exp/decoder_fp16.onnx $model_repo_path/decoder/1/decoder.onnx
	cp $pretrained_model_dir/exp/joiner_fp16.onnx $model_repo_path/joiner/1/joiner.onnx
    else
        cp $pretrained_model_dir/exp/encoder-epoch-9999-avg-1.onnx $model_repo_path/encoder/1/encoder.onnx
        cp $pretrained_model_dir/exp/decoder-epoch-9999-avg-1.onnx $model_repo_path/decoder/1/decoder.onnx
        cp $pretrained_model_dir/exp/joiner-epoch-9999-avg-1.onnx $model_repo_path/joiner/1/joiner.onnx
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    tritonserver --model-repository=$model_repo_path --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000 --http-port 10086
fi
