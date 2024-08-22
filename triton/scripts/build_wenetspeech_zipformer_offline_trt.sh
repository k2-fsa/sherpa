#!/bin/bash
stage=-1
stop_stage=3

export CUDA_VISIBLE_DEVICES=1

pretrained_model_dir=/workspace/icefall-asr-zipformer-wenetspeech-20230615
model_repo_path=./model_repo_offline

# modify model specific parameters according to $pretrained_model_dir/exp/ log files
VOCAB_SIZE=5537

DECODER_CONTEXT_SIZE=2
DECODER_DIM=512
ENCODER_DIM=512  # max(_to_int_tuple(params.encoder_dim)


if [ -d "$pretrained_model_dir/data/lang_char" ] 
then
    echo "pretrained model using char"
    TOKENIZER_FILE=$pretrained_model_dir/data/lang_char
else
    echo "pretrained model using bpe"
    TOKENIZER_FILE=$pretrained_model_dir/data/lang_bpe_500/bpe.model
fi

MAX_BATCH=16
# model instance num
FEATURE_EXTRACTOR_INSTANCE_NUM=2
ENCODER_INSTANCE_NUM=1
JOINER_INSTANCE_NUM=1
DECODER_INSTANCE_NUM=1
SCORER_INSTANCE_NUM=2


icefall_dir=/workspace/icefall
export PYTHONPATH=$PYTHONPATH:$icefall_dir
recipe_dir=$icefall_dir/egs/wenetspeech/ASR/zipformer

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  if [ -d "$pretrained_model_dir" ]
  then
    echo "skip download pretrained model"
  else
    echo "downloading pretrained model"
    cd /workspace
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pkufool/icefall-asr-zipformer-wenetspeech-20230615
    pushd icefall-asr-zipformer-wenetspeech-20230615
    git lfs pull --include "exp/pretrained.pt"
    ln -s ./exp/pretrained.pt ./exp/epoch-9999.pt
    popd
    cd -
  fi
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "export onnx"
    cd ${recipe_dir}
    # WAR: please comment https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py#L1422-L1427 
    # if you would like to use the exported onnx to build trt engine later.
    python3 ./export-onnx.py \
            --tokens $TOKENIZER_FILE/tokens.txt \
            --use-averaged-model 0 \
            --epoch 9999 \
            --avg 1 \
            --exp-dir $pretrained_model_dir/exp/ \
            --num-encoder-layers "2,2,3,4,3,2" \
            --downsampling-factor "1,2,4,8,4,2" \
            --feedforward-dim "512,768,1024,1536,1024,768" \
            --num-heads "4,4,4,8,4,4" \
            --encoder-dim "192,256,384,512,384,256" \
            --query-head-dim 32 \
            --value-head-dim 12 \
            --causal False    || exit 1

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
    cp $pretrained_model_dir/exp/encoder-epoch-9999-avg-1.onnx $model_repo_path/encoder/1/encoder.onnx
    cp $pretrained_model_dir/exp/decoder-epoch-9999-avg-1.onnx $model_repo_path/decoder/1/decoder.onnx
    cp $pretrained_model_dir/exp/joiner-epoch-9999-avg-1.onnx $model_repo_path/joiner/1/joiner.onnx
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   echo "Buiding TRT engine..., skip the stage if you would like to use onnxruntime"
   polygraphy surgeon sanitize $pretrained_model_dir/exp/encoder-epoch-9999-avg-1.onnx --fold-constant -o $pretrained_model_dir/exp/encoder.onnx
   bash scripts/build_trt.sh $MAX_BATCH $pretrained_model_dir/exp/encoder.onnx $model_repo_path/encoder/1/encoder.trt || exit 1

   sed -i "s|onnxruntime|tensorrt|g" $model_repo_path/encoder/config.pbtxt
   sed -i "s|encoder.onnx|encoder.trt|g" $model_repo_path/encoder/config.pbtxt
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    tritonserver --model-repository=$model_repo_path --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000 --http-port 10086
fi
