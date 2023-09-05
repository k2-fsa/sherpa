## Triton Inference Serving Best Practice for Whisper

### Build Image
Directly pull the image we prepared for you or build it from scratch. 
```sh
# using the prepared image
docker pull soar97/triton-whisper:23.06

# build from scratch, cd to the parent dir of Dockerfile.server
docker build . -f Dockerfile.server -t soar97/triton-whisper:23.06
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "whisper-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-whisper:23.06
```

### Export Whisper Model to Onnx
Use our prepared model_repo_whisper from huggingface or prepare it by yourself.
```sh
# using huggingface model_repo_whisper
apt-get install git-lfs
git-lfs install
git clone https://huggingface.co/yuekai/model_repo_whisper_large_v2.git

# prepare it by yourself, inside the whisper-server docker container
bash export-onnx-triton.sh
```

### Launch Server
```sh
# inside the whisper-server docker container
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
pip3 install -r requirement.txt
pip3 install gradio
python3 app.py
```

### Benchmark using Dataset
```sh
git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
cd Triton-ASR-Client
num_task=4
python3 client.py \
    --server-addr localhost \
    --model-name whisper \
    --num-tasks $num_task \
    --whisper-prompt "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --manifest-dir ./datasets/mini_en
```

### Benchmark Results
Decoding on a single V100 GPU, audios are padded to 30s, using aishell1 test set files

[Details](media/stats_summary_op14_single_batch_large-v2.txt)
| Model | Backend   | Concurrency | RTF     |
|-------|-----------|-----------------------|---------|
| Large-v2 | ONNX FP16 | 4                   | 0.14 |

|Module| Time Distribution|
|--|--|
|feature_extractor|0.8%|
|encoder|9.6%|
|decoder|67.4%|
|greedy search|22.2%|