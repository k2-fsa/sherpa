export CUDA_VISIBLE_DEVICES="0"

model_repo=./model_repo_whisper_qwen_trtllm_exp

tritonserver --model-repository $model_repo