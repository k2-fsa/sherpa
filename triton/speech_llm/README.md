## Triton Inference Serving Best Practice for Speech LLM

### Model Training
See https://github.com/k2-fsa/icefall/tree/master/egs/speech_llm/ASR_LLM. 

### Quick Start
Directly launch the service using docker compose.
```sh
docker compose up --build
```

### Build Image
Build the docker image from scratch. 
```sh
# build from scratch, cd to the parent dir of Dockerfile.server
docker build . -f Dockerfile.server -t soar97/triton-whisper-qwen:24.08
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "whisper-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-whisper-qwen:24.08
```

### Export Models to TensorRT-LLM
Inside docker container, we would follow the official guide of TensorRT-LLM to build qwen and whisper TensorRT-LLM engines. See [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper).

```sh
bash build.sh
```

### Launch Server
```sh
bash launch_server.sh
```

<!-- ### Launch Gradio WebUI Client
The gradio client supports text and speech as the inputs.

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
python3 client.py \
    --server-addr localhost \
    --model-name infer_bls \
    --num-tasks $num_task \
    --manifest-dir ./datasets/aishell1_test \
    --compute-cer
```

### Benchmark Results
Decoding on a single A10 GPU, audios without padding, using aishell1 test set files

| Model | Backend   | Concurrency | RTFx     | RTF | 
|-------|-----------|-----------------------|---------|--|
| Whisper Large-v2 Encoder + Qwen 1.5B | python backend speech encoder + trt-llm backend llm | 8                   | 156 | 0.0064|