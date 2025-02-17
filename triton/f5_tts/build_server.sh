


vocoder=$1
model_repo=./model_repo
F5_TTS_HF_DOWNLOAD_PATH=./F5-TTS
F5_TTS_TRT_LLM_CHECKPOINT_PATH=./trtllm_ckpt
F5_TTS_TRT_LLM_ENGINE_PATH=./f5_trt_llm_engine

python_package_path=/usr/local/lib/python3.12/dist-packages



DIR=$python_package_path/tensorrt_llm/models/f5tts
if [ -d "$DIR" ]; then
  echo "Directory '$DIR' exists."
else
  echo "Directory '$DIR' does not exist."
  git clone https://github.com/yuekaizhang/TensorRT-LLM.git -b f5
  cp -r TensorRT-LLM/tensorrt_llm/models/* $python_package_path/tensorrt_llm/models
fi

huggingface-cli download SWivid/F5-TTS --local-dir $F5_TTS_HF_DOWNLOAD_PATH
python3 ./scripts/convert_checkpoint.py \
        --timm_ckpt "$F5_TTS_HF_DOWNLOAD_PATH/F5TTS_Base/model_1200000.pt" \
        --output_dir "$F5_TTS_TRT_LLM_CHECKPOINT_PATH"
trtllm-build --checkpoint_dir $F5_TTS_TRT_LLM_CHECKPOINT_PATH --output_dir $F5_TTS_TRT_LLM_ENGINE_PATH

pip install vocos
python3 scripts/export_vocoder_to_onnx.py --vocoder $vocoder
bash scripts/export_vocos_trt.sh


cp -r ./model_repo_f5_tts $model_repo
python3 fill_template.py -i $model_repo/f5_tts/config.pbtxt vocab:$F5_TTS_HF_DOWNLOAD_PATH/F5TTS_Base/vocab.txt,model:$F5_TTS_TRT_LLM_ENGINE_PATH/F5TTS_Base/model_1200000.pt,trtllm:$F5_TTS_TRT_LLM_ENGINE_PATH,vocoder:$1
cp vocos_vocoder.plan $model_repo/vocoder/1/vocoder.plan

tritonserver --model-repository=$model_repo