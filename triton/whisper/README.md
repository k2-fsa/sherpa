## Triton Inference Serving Best Practice for Whisper TensorRT-LLM

### Quick Start
Directly launch the service using docker compose.
```sh
# MODEL_IDs=("large-v3-turbo" "large-v3" "large-v2-turbo-multi-hans" "large-v2-multi-hans")
MODEL_ID=large-v3-turbo docker compose up
```

### Build Image
Build the docker image from scratch. 
```sh
# build from scratch, cd to the parent dir of Dockerfile.server
docker build . -f Dockerfile.server -t soar97/triton-whisper:24.09
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "whisper-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-whisper:24.09
```

### Export Whisper Model to TensorRT-LLM
Inside docker container, we would follow the official guide of TensorRT-LLM to build whisper TensorRT-LLM engines. See [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper).

```sh
# We already have a clone of TensorRT-LLM inside container, so no need to clone it.
cd TensorRT-LLM/examples/whisper

# take large-v3 model as an example
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt

INFERENCE_PRECISION=float16
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
checkpoint_dir=tllm_checkpoint
output_dir=whisper_large_v3

# Convert the large-v3 openai model into trtllm compatible checkpoint.
python3 convert_checkpoint.py \
                --output_dir $checkpoint_dir

# Build the large-v3 trtllm engines
trtllm-build --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --moe_plugin disable \
              --enable_xqa disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --max_input_len 3000 --max_seq_len=3000


trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --enable_xqa disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}

# prepare the model_repo_whisper
cd sherpa/triton/whisper
model_repo=model_repo_whisper
rm -rf $model_repo
cp model_repo_whisper_trtllm $model_repo -r
wget --directory-prefix=$model_repo/infer_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=$model_repo/whisper/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

output_dir=/workspace/TensorRT-LLM/examples/whisper/whisper_turbo
n_mels=128
zero_pad=false

TRITON_MAX_BATCH_SIZE=64
MAX_QUEUE_DELAY_MICROSECONDS=100
python3 fill_template.py -i $model_repo/whisper/config.pbtxt engine_dir:${output_dir},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
python3 fill_template.py -i $model_repo/infer_bls/config.pbtxt engine_dir:${output_dir},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
```
### Using Fine-tuned Whisper
Official whisper models only accept 30-second audios. To improve the throughput, you could fine-tune the whisper model to remove the 30 seconds restriction. See [examples](https://github.com/k2-fsa/icefall/blob/master/egs/aishell/ASR/whisper/whisper_encoder_forward_monkey_patch.py#L15). 

We prepared two [Chinese fine-tuned whisper](https://github.com/k2-fsa/icefall/blob/master/egs/multi_zh-hans/ASR/RESULTS.md#multi-chinese-datasets-without-datatang-200h-finetuning-results-on-whisper-large-v2) TensorRT-LLM weights repo. They could be directly used from [here.](https://huggingface.co/yuekai/whisper_multi_zh_tllm_checkpoint/tree/main)

### Launch Server
Log of directory tree:
```sh
model_repo_whisper_trtllm
├── infer_bls
│   ├── 1
│   │   ├── model.py
│   │   ├── multilingual.tiktoken
│   │   └── tokenizer.py
│   └── config.pbtxt
└── whisper
    ├── 1
    │   ├── fbank.py
    │   ├── mel_filters.npz
    │   └── model.py
    └── config.pbtxt

4 directories, 8 files
```
```sh
# launch the server
python3 launch_triton_server.py --world_size 1 --model_repo=$model_repo/ --tensorrt_llm_model_name whisper,infer_bls --multimodal_gpu0_cuda_mem_pool_bytes 300000000
```

<!-- ### Launch Gradio WebUI Client
The gradio client supports text as the input, which enables users to prompt the whisper model.

See [Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization](https://arxiv.org/abs/2305.11095) for more details.

![Demo](media/Screenshot.jpg)

```sh
git-lfs install
git clone https://huggingface.co/spaces/yuekai/triton-asr-client.git
cd triton-asr-client
pip3 install -r requirements.txt
python3 app.py
``` -->

### Benchmark using Dataset
```sh
git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
cd Triton-ASR-Client
num_task=16
dataset=aishell1_test
python3 client.py \
    --server-addr localhost \
    --model-name infer_bls \
    --num-tasks $num_task \
    --text-prompt "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --manifest-dir ./datasets/$dataset \
    --log-dir ./log_sherpa_multi_hans_whisper_large_ifb_$num_task \
    --compute-cer
```

<!-- ### Benchmark Results
Decoding on a single V100 GPU, audios are padded to 30s, using aishell1 test set files

| Model | Backend   | Concurrency | RTF     |
|-------|-----------|-----------------------|---------|
| Large-v2 | ONNX FP16 (deprecated) | 4                   | 0.14 |
| Large-v3 | TensorRT-LLM FP16 | 4                   | 0.0209 | -->