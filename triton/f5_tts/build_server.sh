

stage=$1
stop_stage=$2
echo "Start stage: $stage, Stop stage: $stop_stage"
export CUDA_VISIBLE_DEVICES=0
# vocoder=$1
vocoder=vocos
echo $vocoder || exit 1

f5_trt_llm_local_dir=/home/scratch.yuekaiz_wwfo_1/tts/TensorRT-LLM
# prepare for f5 tts trtllm
python_package_path=/usr/local/lib/python3.12/dist-packages
mkdir -p $python_package_path/tensorrt_llm/models/f5tts
# wget https://raw.githubusercontent.com/yuekaizhang/TensorRT-LLM/f5/tensorrt_llm/models/__init__.py -O $python_package_path/tensorrt_llm/models/__init__.py
# wget https://raw.githubusercontent.com/yuekaizhang/TensorRT-LLM/f5/tensorrt_llm/models/f5tts/model.py -O $python_package_path/tensorrt_llm/models/f5tts/model.py
# wget https://raw.githubusercontent.com/yuekaizhang/TensorRT-LLM/f5/tensorrt_llm/models/f5tts/modules.py -O $python_package_path/tensorrt_llm/models/f5tts/modules.py
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Copying f5 tts trtllm files"
    cp $f5_trt_llm_local_dir/tensorrt_llm/models/__init__.py $python_package_path/tensorrt_llm/models/__init__.py
    cp $f5_trt_llm_local_dir/tensorrt_llm/models/f5tts/model.py $python_package_path/tensorrt_llm/models/f5tts/model.py
    cp $f5_trt_llm_local_dir/tensorrt_llm/models/f5tts/modules.py $python_package_path/tensorrt_llm/models/f5tts/modules.py
fi
# export f5 tts trtllm
# F5_TTS_HF_DOWNLOAD_PATH=./F5-TTS
# F5_TTS_TRT_LLM_CHECKPOINT_PATH=./trtllm_ckpt
# F5_TTS_TRT_LLM_ENGINE_PATH=./f5_trt_llm_engine

F5_TTS_HF_DOWNLOAD_PATH=/home/scratch.yuekaiz_wwfo_1/tts/F5_TTS_Faster/F5_TTS
F5_TTS_TRT_LLM_CHECKPOINT_PATH=/home/scratch.yuekaiz_wwfo_1/tts/tmp/trtllm_ckpt_f5
F5_TTS_TRT_LLM_ENGINE_PATH=/home/scratch.yuekaiz_wwfo_1/tts/tmp/f5_trt_llm_engine

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    echo "Downloading f5 tts from huggingface"
    trtllm-build --checkpoint_dir $F5_TTS_TRT_LLM_CHECKPOINT_PATH --output_dir $F5_TTS_TRT_LLM_ENGINE_PATH --max_batch_size 16 --remove_input_padding disable
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Downloading f5 tts from huggingface"
    # huggingface-cli download SWivid/F5-TTS --local-dir $F5_TTS_HF_DOWNLOAD_PATH
    # python3 ./scripts/convert_checkpoint.py \
    #     --timm_ckpt "$F5_TTS_HF_DOWNLOAD_PATH/F5TTS_Base/model_1200000.pt" \
    #     --output_dir "$F5_TTS_TRT_LLM_CHECKPOINT_PATH"
    trtllm-build --checkpoint_dir $F5_TTS_TRT_LLM_CHECKPOINT_PATH \
      --max_batch_size 16 \
      --output_dir $F5_TTS_TRT_LLM_ENGINE_PATH --bert_attention_plugin disable
fi


# export vocos vocoder
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Exporting vocos vocoder"
    pip install vocos
    python3 scripts/export_vocoder_to_onnx.py --vocoder $vocoder
    bash scripts/export_vocos_trt.sh
fi
model_repo=./model_repo
# build triton server
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Building triton server"

    rm -r $model_repo
    cp -r ./model_repo_f5_tts $model_repo
python3 fill_template.py -i $model_repo/f5_tts/config.pbtxt vocab:$F5_TTS_HF_DOWNLOAD_PATH/F5TTS_Base/vocab.txt,model:$F5_TTS_HF_DOWNLOAD_PATH/F5TTS_Base/model_1200000.pt,trtllm:$F5_TTS_TRT_LLM_ENGINE_PATH,vocoder:$vocoder
    cp vocos_vocoder.plan $model_repo/vocoder/1/vocoder.plan
fi

# start triton server
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Starting triton server"
    tritonserver --model-repository=$model_repo
fi

num_task=1
log_dir=./offline_log_${num_task}_test_input_lengths_bert_plugin_test
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Testing triton server"
    python3 client.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts --log-dir $log_dir
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    bash scripts/compute_wer.sh $log_dir wenetspeech4tts
fi



if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    # pip install vocos
    torchrun --nproc_per_node=1 \
    run.py --output-dir $log_dir \
    --batch-size $num_task \
    --model-path $F5_TTS_HF_DOWNLOAD_PATH/F5TTS_Base/model_1200000.pt \
    --vocab-file $F5_TTS_HF_DOWNLOAD_PATH/F5TTS_Base/vocab.txt \
    --tllm-model-dir $F5_TTS_TRT_LLM_ENGINE_PATH
fi