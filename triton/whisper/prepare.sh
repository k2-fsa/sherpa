

output_dir=/workspace/whisper_turbo_and_distill_tllm_checkpoint/whisper_turbo
n_mels=128
zero_pad=false

# output_dir=/workspace/whisper_multi_zh_tllm_checkpoint/distil_whisper_multi_zh_remove_padding
# n_mels=80
# zero_pad=true

model_repo=model_repo_whisper
rm -rf $model_repo
cp model_repo_whisper_trtllm $model_repo -r
wget --directory-prefix=$model_repo/infer_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=$model_repo/whisper/1 assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

TRITON_MAX_BATCH_SIZE=64
MAX_QUEUE_DELAY_MICROSECONDS=100
python3 fill_template.py -i $model_repo/whisper/config.pbtxt engine_dir:${output_dir},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
python3 fill_template.py -i $model_repo/infer_bls/config.pbtxt engine_dir:${output_dir},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}

python3 launch_triton_server.py --world_size 1 --model_repo=$model_repo/ --tensorrt_llm_model_name whisper,infer_bls --multimodal_gpu0_cuda_mem_pool_bytes 300000000