# Inference Serving Best Practice for Transducer ASR based on Icefall <!-- omit in toc -->

In this tutorial, we'll go through how to run  non-streaming (offline) and streaming ASR Transducer models trained by [Icefall](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless3) **on GPUs**, and deploy it as service with NVIDIA [Triton Inference Server](https://github.com/triton-inference-server/server).
## Table of Contents <!-- omit in toc -->

- [Preparation](#preparation)
  - [Prepare Environment](#prepare-environment)
- [Deploy on Triton Inference Server](#deploy-on-triton-inference-server)
  - [Quick Start](#quick-start)
- [Inference Client](client/README.md)


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
