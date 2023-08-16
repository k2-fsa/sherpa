# Inference Serving Best Practice for Transducer ASR based on Icefall <!-- omit in toc -->

In this tutorial, we'll go through how to run  non-streaming (offline) and streaming ASR Transducer models trained by [Icefall](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless3) **on GPUs**, and deploy it as service with NVIDIA [Triton Inference Server](https://github.com/triton-inference-server/server).
## Table of Contents <!-- omit in toc -->

- [Preparation](#preparation)
  - [Prepare Environment](#prepare-environment)
- [Deploy on Triton Inference Server](#deploy-on-triton-inference-server)
  - [Quick Start](#quick-start)
- [Inference Client](client/README.md)
- [Using TensorRT acceleration](#using-tensorrt-acceleration)
  - [TRT Quick start](#trt-quick-start)
  - [Benchmark for Conformer TRT encoder vs ONNX](#benchmark-for-conformer-trt-encoder-vs-onnx)


## Preparation

First of all, we need to get environment, models ready.

### Prepare Environment

Clone the repository:

```bash
# Clone Sherpa repo
git clone https://github.com/k2-fsa/sherpa.git
cd sherpa
export SHERPA_SRC=$PWD
```
We highly recommend you to use docker containers to save your life.

Build the server docker image:
```
cd $SHERPA_SRC/triton
docker build . -f Dockerfile/Dockerfile.server -t sherpa_triton_server:latest --network host
```
Alternatively, you could directly pull the pre-built image based on tritonserver 22.12.
```
docker pull soar97/triton-k2:22.12.1
```

If you are planning to use TRT to accelerate the inference speed, you can use the following prebuit image:
```
docker pull wd929/sherpa_wend_23.04:v1.1
```

Start the docker container:
```bash
docker run --gpus all -v $SHERPA_SRC:/workspace/sherpa --name sherpa_server --net host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it soar97/triton-k2:22.12.1
```
Now, you should enter into the container successfully.


## Deploy on Triton Inference Server

In this part, we'll go through how to deploy the model on Triton.

The model repositories are provided in `model_repo_offline` and `model_repo_streaming` directory, you can find directories standing for each of the components. And there is a `transducer` dir which ensembles all the components into a whole pipeline. Each of those component directories contains a config file `config.pbtxt` and a version directory containing the model file.  

### Quick Start

Now start server:

```bash
# Inside the docker container
# If you want to use greedy search decoding
cd /Your_SHERPA_SRC/triton/
apt-get install git-lfs
pip3 install -r ./requirements.txt
export CUDA_VISIBLE_DEVICES="your_gpu_id"

bash scripts/build_wenetspeech_pruned_transducer_stateless5_streaming.sh
bash scripts/build_librispeech_pruned_transducer_stateless3_streaming.sh
```

## Using TensorRT acceleration

### TRT Quick start

You can directly use the following script to export TRT engine and start Triton server for Conformer Offline model: 

```bash
bash scripts/build_librispeech_pruned_transducer_stateless3_offline_trt.sh
```

### Export to TensorRT Step by Step

If you want to build TensorRT for your own model, you can try the following steps:

#### Preparation for TRT

First of all, you have to install the TensorRT. Here we suggest you to use docker container to run TRT. Just run the following command:

```bash
docker run --gpus '"device=0"' -it --rm --net host -v $PWD/:/k2 nvcr.io/nvidia/tensorrt:23.04-py3
```
You can also see [here](https://github.com/NVIDIA/TensorRT#build) to build TRT on your machine. 

#### Model export 

You have to prepare the ONNX model by referring [here](https://github.com/k2-fsa/sherpa/blob/master/triton/scripts/build_librispeech_pruned_transducer_stateless3_offline.sh#L41C1-L41C1) to export your models into ONNX format. Assume you have put your ONNX model in the `$model_dir` directory. 
Then, just run the command:

```bash
bash scripts/build_trt.sh 128 $model_dir/encoder.onnx model_repo_offline/encoder/1/encoder.trt
```

The generated TRT model will be saved into `model_repo_offline/encoder/1/encoder.trt`. 
Then you can start the Triton server.


### Benchmark for Conformer TRT encoder vs ONNX

| Model  | Batch size| Avg latency(ms) |   QPS    |
|--------|-----------|-----------------|----------|
| ONNX   |     1     |     7.44        |  134.48  |
|        |     8     |     14.92       |  536.09  |
|        |    16     |     22.84       |  700.67  |
|        |    32     |     41.62       |  768.84  |
|        |    64     |     80.48       |  795.27  |
|        |   128     |     171.97      |  744.32  |
|  TRT   |     1     |     5.21834     |  193.93  |
|        |     8     |     11.7826     |  703.49  |
|        |    16     |     20.4444     |  815.79  |
|        |    32     |     37.583      |  893.56  |
|        |    64     |     69.8312     |  965.40  |
|        |   128     |     139.702     |  964.57  |
