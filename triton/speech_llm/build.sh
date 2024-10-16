huggingface_checkpoint_dir=./whisper_qwen_1.5B
repo=yuekai/whisper_qwen_multi_hans_zh_triton_checkpoint
adapter_dir=$huggingface_checkpoint_dir/icefall_asr_multi-hans_whisper_qwen2_1.5B/epoch-2-avg-6.pt
engine_path=$huggingface_checkpoint_dir/qwen2_1.5B_instruct_fp16_merged
# commands for whisper_qwen_7B
# huggingface_checkpoint_dir=./whisper_qwen_7B
# repo=yuekai/whisper_qwen_7b_multi_hans_zh_triton_checkpoint
# adapter_dir=$huggingface_checkpoint_dir/icefall_asr_multi-hans_whisper_qwen2_7B/epoch-999.pt
# engine_path=$huggingface_checkpoint_dir/qwen2_7B_instruct_int8_woq_merged

huggingface-cli download --local-dir $huggingface_checkpoint_dir $repo
cd $huggingface_checkpoint_dir && bash build_qwen.sh && bash build_whisper_encoder.sh && cd -

model_repo=./model_repo_whisper_qwen_trtllm_exp
rm -rf $model_repo
cp -r ./model_repo_whisper_qwen_trtllm $model_repo || exit 1


encoder_engine_dir=$huggingface_checkpoint_dir/whisper_multi_zh

max_batch=16
decoupled_mode=false
max_queue_delay_microseconds=0
n_mels=80
n_instances=8
python3 fill_template.py -i $model_repo/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:$max_batch,decoupled_mode:${decoupled_mode},max_beam_width:1,engine_dir:${engine_path},max_tokens_in_paged_kv_cache:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${max_queue_delay_microseconds}
python3 fill_template.py -i $model_repo/speech_encoder/config.pbtxt triton_max_batch_size:$max_batch,n_mels:$n_mels,adapter_dir:$adapter_dir,encoder_engine_dir:$encoder_engine_dir,max_queue_delay_microseconds:${max_queue_delay_microseconds}
python3 fill_template.py -i $model_repo/infer_bls/config.pbtxt triton_max_batch_size:$max_batch,n_instances:$n_instances,decoupled_mode:${decoupled_mode},max_queue_delay_microseconds:${max_queue_delay_microseconds}