# Download Models https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17
declare -A MODELS=(
    ["large-v3"]="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
    ["large-v3-turbo"]="https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt"
    ["large-v2-multi-hans"]="https://huggingface.co/yuekai/icefall_asr_multi-hans-zh_whisper/resolve/main/v1.1/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt"
    ["large-v2-turbo-multi-hans"]="https://huggingface.co/yuekai/icefall_asr_multi-hans-zh_whisper/resolve/main/v1-distill/distill-whisper-large-v2-multi-hans-epoch-6-avg-8.pt"
)

build_model() {
    local model_id=$1
    local checkpoint_dir=$2
    local output_dir=$3

    local URL=${MODELS[$model_id]}
    
    echo "Downloading $MODEL_ID from $URL..."
    wget -nc "$URL"

    echo "Converting checkpoint for model: $model_id"
    python3 convert_checkpoint.py \
        --output_dir "$checkpoint_dir" \
        --model_path "$(basename $URL)"

    local INFERENCE_PRECISION=float16
    local MAX_BEAM_WIDTH=4
    local MAX_BATCH_SIZE=64

    echo "Building encoder for model: $model_id"
    trtllm-build --checkpoint_dir "${checkpoint_dir}/encoder" \
                  --output_dir "${output_dir}/encoder" \
                  --moe_plugin disable \
                  --enable_xqa disable \
                  --max_batch_size "$MAX_BATCH_SIZE" \
                  --gemm_plugin disable \
                  --bert_attention_plugin "$INFERENCE_PRECISION" \
                  --max_input_len 3000 --max_seq_len 3000

    echo "Building decoder for model: $model_id"
    trtllm-build --checkpoint_dir "${checkpoint_dir}/decoder" \
                  --output_dir "${output_dir}/decoder" \
                  --moe_plugin disable \
                  --enable_xqa disable \
                  --max_beam_width "$MAX_BEAM_WIDTH" \
                  --max_batch_size "$MAX_BATCH_SIZE" \
                  --max_seq_len 114 \
                  --max_input_len 14 \
                  --max_encoder_input_len 3000 \
                  --gemm_plugin "$INFERENCE_PRECISION" \
                  --bert_attention_plugin "$INFERENCE_PRECISION" \
                  --gpt_attention_plugin "$INFERENCE_PRECISION"
}

launch_triton_repo() {
    local output_dir=$1
    n_mels=$(cat ${output_dir}/encoder/config.json | grep n_mels | awk -F': ' '{print $2}' | tr -d ',')
    if [[ "$output_dir" == *"multi-hans"* ]]; then
        zero_pad=true # fine-tuned model could remove 30s padding, so set pad to none
    else
        zero_pad=false
    fi

    echo "output_dir: $output_dir", "n_mels: $n_mels", "zero_pad: $zero_pad"

    model_repo=model_repo_whisper
    rm -rf $model_repo
    cp model_repo_whisper_trtllm $model_repo -r
    wget -nc --directory-prefix=$model_repo/infer_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
    wget -nc --directory-prefix=$model_repo/whisper/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

    TRITON_MAX_BATCH_SIZE=64
    MAX_QUEUE_DELAY_MICROSECONDS=100
    python3 fill_template.py -i $model_repo/whisper/config.pbtxt engine_dir:${output_dir},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 fill_template.py -i $model_repo/infer_bls/config.pbtxt engine_dir:${output_dir},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}

    python3 launch_triton_server.py --world_size 1 --model_repo=$model_repo/ --tensorrt_llm_model_name whisper,infer_bls --multimodal_gpu0_cuda_mem_pool_bytes 300000000
}


MODEL_IDs=("large-v3-turbo" "large-v3" "large-v2-turbo-multi-hans" "large-v2-multi-hans")
CUDA_VISIBLE_DEVICES=0

model_id=$1
checkpoint_dir="${model_id}_tllm_checkpoint"
output_dir="whisper_${model_id}"

if printf '%s\n' "${MODEL_IDs[@]}" | grep -q "^$model_id$"; then
    build_model $model_id "$checkpoint_dir" "$output_dir" || exit 1
    launch_triton_repo "$output_dir" || exit 1
else
    echo "$model_id is NOT in the MODEL_IDs array."
    exit 1
fi

