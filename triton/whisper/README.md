## Triton Inference Serving Best Practice for Whisper TensorRT-LLM

### Quick Start
Directly launch the service using docker compose.
```sh
docker compose up --build
```

### Build Image
Build the docker image from scratch. 
```sh
# build from scratch, cd to the parent dir of Dockerfile.server
docker build . -f Dockerfile.server -t soar97/triton-whisper:24.05
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "whisper-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-whisper:24.05
```

### Export Whisper Model to TensorRT-LLM
Inside docker container, we would follow the official guide of TensorRT-LLM to build whisper TensorRT-LLM engines. See [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper).

```sh
# We already have a clone of TensorRT-LLM inside container, so no need to clone it.
cd /workspace/TensorRT-LLM/examples/whisper

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
                --paged_kv_cache disable \
                --moe_plugin disable \
                --enable_xqa disable \
                --use_custom_all_reduce disable \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --gemm_plugin disable \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable

trtllm-build --checkpoint_dir ${checkpoint_dir}/decoder \
                --output_dir ${output_dir}/decoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --enable_xqa disable \
                --use_custom_all_reduce disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --max_output_len 100 \
                --max_input_len 14 \
                --max_encoder_input_len 1500 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable

# prepare the model_repo_whisper_trtllm
cd sherpa/triton/whisper
ln -sv /workspace/TensorRT-LLM/examples/whisper/whisper_large_v3 ./model_repo_whisper_trtllm/whisper/1/
wget --directory-prefix=./model_repo_whisper_trtllm/whisper/1/ https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=./model_repo_whisper_trtllm/whisper/1/ assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
```

### Launch Server
Log of directory tree:
```sh
model_repo_whisper_trtllm
└── whisper
    ├── 1
    │   ├── fbank.py
    │   ├── mel_filters.npz
    │   ├── model.py
    │   ├── multilingual.tiktoken
    │   ├── tokenizer.py
    │   ├── whisper_large_v3 -> /workspace/TensorRT-LLM/examples/whisper/whisper_large_v3
    │   └── whisper_trtllm.py
    └── config.pbtxt

3 directories, 7 files
```
```sh
bash launch_server.sh
```

### Launch Gradio WebUI Client
The gradio client supports text as the input, which enables users to prompt the whisper model.

See [Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization](https://arxiv.org/abs/2305.11095) for more details.

![Demo](media/Screenshot.jpg)

```sh
git-lfs install
git clone https://huggingface.co/spaces/yuekai/triton-asr-client.git
cd triton-asr-client
pip3 install -r requirements.txt
python3 app.py
```

### Benchmark using Dataset
```sh
git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
cd Triton-ASR-Client
num_task=16
python3 client.py \
    --server-addr localhost \
    --model-name whisper \
    --num-tasks $num_task \
    --whisper-prompt "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --manifest-dir ./datasets/aishell1_test
```

### Benchmark Results
Decoding on a single V100 GPU, audios are padded to 30s, using aishell1 test set files

| Model | Backend   | Concurrency | RTF     |
|-------|-----------|-----------------------|---------|
| Large-v2 | ONNX FP16 (deprecated) | 4                   | 0.14 |
| Large-v3 | TensorRT-LLM FP16 | 4                   | 0.0209 |