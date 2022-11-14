# Inference Serving Best Practice for Transducer ASR based on Icefall <!-- omit in toc -->

In this tutorial, we'll go through how to run  non-streaming (offline) and streaming ASR Transducer models trained by [Icefall](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless3) **on GPUs**, and deploy it as service with NVIDIA [Triton Inference Server](https://github.com/triton-inference-server/server).
## Table of Contents <!-- omit in toc -->

- [Preparation](#preparation)
  - [Prepare Environment](#prepare-environment)
  - [Prepare Models](#prepare-models)
- [Deploy on Triton Inference Server](#deploy-on-triton-inference-server)
  - [Quick Start](#quick-start)
  - [Advanced](#advanced)
    - [Specify which GPUs for deployment](#specify-which-gpus-for-deployment)
    - [Set the number of model instances per GPU](#set-the-number-of-model-instances-per-gpu)
    - [Set the ports exposed by server](#set-the-ports-exposed-by-server)
    - [Set the request scheduler](#set-the-request-scheduler)
- [Inference Client](#inference-client)
  - [Quick Start](#quick-start-1)
  - [Performance Test](#performance-test)
- [Benchmarks](#benchmarks)
  - [FP32 Performance on Single GPU](#fp32-performance-on-single-gpu)
    - [FP32 Performance of Small Model for Librispeech](#fp32-performance-of-small-model-for-librispeech)
  - [Reference Accuracy](#reference-accuracy)

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
# It may take a lot of time since we build k2 from source.
docker build . -f Dockerfile/Dockerfile.server -t sherpa_triton_server:latest --network host
```
Start the docker container:
```bash
docker run --gpus all -v $SHERPA_SRC:/workspace/sherpa --name sherpa_server --net host --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -it sherpa_triton_server:latest
```
Now, you should enter into the container successfully.

# Prepare pretrained models

In this section, we would take jit export as an example for offline ASR and use onnx export as an example for streaming ASR.

Offline Model Export:
```bash
git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-04-29

# export them to three jit models: encoder_jit.pt, decoder_jit.pt, joiner_jit.pt
cp $SHERPA/triton/scripts/conformer_triton.py $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3/
cp $SHERPA/triton/scripts/export_jit.py $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3/

cd $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3
python3 export_jit.py --pretrained-model <pretrained_model_path> --output-dir <jit_model_dir> --bpe-model <bpe_model_path>

# copy bpe file to <jit_model_dir>, later we would mount <jit_model_dir> to the triton docker container
cp <bpe_model_path> <jit_model_dir>
```

Streaming Model Export:
```bash
cp $SHERPA/triton/scripts/*onnx*.py $ICEFALL_DIR/egs/librispeech/ASR/pruned_stateless_transducer3/

cd $ICEFALL_DIR/egs/librispeech/ASR/

git clone https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625

ln -s ./icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/exp/pretrained* ./icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/exp/epoch-999.pt 

./pruned_transducer_stateless3/export_onnx.py \
    --exp-dir ./icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/exp \
    --bpe-model ./icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625/data/lang_bpe_500/bpe.model \
    --epoch 999 \
    --avg 1 \
    --streaming-model 1 \
    --causal-convolution 1 \
    --onnx 1 \
    --left-context 64 \
    --right-context 4 \
    --fp16
```


## Deploy on Triton Inference Server

Now we have exported the pretrained ASR model, then we need to consider how to deploy the model on the server as an ASR service, to allow users to send audio requests and get recognition results. Actually, [Triton Inference Server](https://github.com/triton-inference-server/server) does the most of serving work for us, it handles requests/results sending and receiving, request scheduling, load balance, and inference execution.  In this part, we'll go through how to deploy the model on Triton.

The model repositories are provided in `model_repo_offline` and `model_repo_streaming` directory, you can find directories standing for each of the components. And there is a `conformer_transducer` dir which ensembles all the components into a whole pipeline. Each of those component directories contains a config file `config.pbtxt` and a version directory containing the model file. However, the version directories of encoder and decoder are still empty since we have not put the exported models into them.  

### Quick Start

Now start server:

```bash
# Inside the docker container
# If you want to use greedy search decoding
bash /workspace/scripts/start_streaming(offline)_server(_jit).sh

# Or if you want to use fast beam search decoding
bash /workspace/scripts/start_streaming(offline)_server_fast_beam.sh
```

If you meet any issues during the process, please file an issue.

### Advanced

Here we introduce some advanced configuration/options for deploying the ASR server.

#### Deploy onnx with arbitrary pruned_transducer_stateless_X(2,3,4,5) model for Chinese or English recipes 
```bash
# e.g. pretrained_model_dir=/workspace/icefall/egs/wenetspeech/ASR/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming

# Modify model hyper parameters according to $pretrained_model_dir/exp/onnx_export.log
# Then,
bash scripts/build.sh
```

#### Specify which GPUs for deployment

If you have multiple GPUs on the server machine, you can specify which GPUs will be used for deploying ASR service. To do so, just change the `-e CUDA_VISIBLE_DEVICES=` option or just use to specify which GPU to make visible in the container when starting server container. 

For example, if you just want to use GPU 1, 2, 3 for deployment, then use the following options to start the server:

```bash
docker run --gpus '"device=1,2,3"' -v $PWD/model_repo:/ws/model_repo_offline_jit -v <jit_model_dir>:/ws/jit_model/ --name sherpa_server --net host --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -it sherpa_triton_server:latest
```

#### Set the number of model instances per GPU

Triton can provide multiple [instances of a model](https://github.com/triton-inference-server/server/blob/master/docs/architecture.md#concurrent-model-execution) so that multiple inference requests for that model can be handled simultaneously. You can set the number of model instances on each GPU by modifying the `config.pbtxt` file of the any of the component model in the `model_repo`, as [Triton document](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#instance-groups) says.

For example, if you want to set 4 model instances for encoder component on each GPU (given that each GPU memory is large enough to handle to instances), just edit the following lines of config file model_repo/encoder/config.pbtxt`:

```
instance_group [
  {
    count: 4
    kind: KIND_GPU
  }
]
```

Elsewise, if you want to set only 1 model instance on each GPU, just change the `count:` field to `1`.

You can also specify which GPUs to use for deployment, and how many model instances running on those GPUs. For example, you can set 1 execution instance on GPU 0, 2 execution instances on GPU 1 and 2:

```
instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    },
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 1, 2 ]
    }
  ]
```

#### Set the ports exposed by server

The default ports exposed by server to clients is: port 8000 for HTTP inference service; port 8001 for gRPC inference service; port 8002 for Metrics service. If the default ports are occupied by other services, you can change the ports that are exposed to the clients, by specify `-p` option when starting the server.

For example, if you want to set port 8003 for HTTP inference service, 8004 for gRPC inference service, and 8005 for Metrics service, then use the following command to start the server:

```bash
docker run --gpus all -v $PWD/model_repo:/ws/model_repo -v <jit_model_dir>:/ws/jit_model/ --name sherpa_server --net host --shm-size=1g --ulimit memlock=-1 -p 8003:8003 -p 8004:8004 -p 8005:8005 --ulimit stack=67108864 -it sherpa_triton_server:latest
```

And then add the following options when start the Triton server:

```bash
tritonserver --model-repository=/ws/model_repo --http-port=8003 --grpc-port=8004 --metrics-port=8005
```

Please note that: if you change the exposed ports of server, you should also specify the same ports when sending requests via client program. For how to specify server port for sending requests.

#### Set the request scheduler

With Triton, you can choose various scheduler modes and batching strategies for coming requests, as described in [Triton document](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#scheduling-and-batching). In our project, we use [dynamic batcher](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher) by default, which allows inference requests to be combined by the server, so that a batch is created dynamically.

You can change the settings of the dynamic batcher by editting `config.pbtxt` file of each model:

```
dynamic_batching {
    preferred_batch_size: [8, 16, 32, 64]
    max_queue_delay_microseconds: 10000
}
```

The [preferred_batch_size](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#preferred-batch-sizes) property indicates the batch sizes that the dynamic batcher should attempt to create. For example, the above configuration enables dynamic batching with preferred batch sizes of 8, 16, 32 and 64.

The `max_queue_delay_microseconds` property setting changes the dynamic batcher behavior when a batch of a preferred size cannot be created. When a batch of a preferred size cannot be created from the available requests, the dynamic batcher will delay sending the batch as long as no request is delayed longer than the configured max_queue_delay_microseconds value. If a new request arrives during this delay and allows the dynamic batcher to form a batch of a preferred batch size, then that batch is sent immediately for inferencing. If the delay expires the dynamic batcher sends the batch as is, even though it is not a preferred size.

## Inference Client

In this section, we will show how to send requests to our deployed non-streaming ASR service, and receive the recognition results. Also, we can use client to test the accuracy of the ASR service on a test dataset. In addition, we can use [perf_analyzer](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md) provided by Triton to test the performance of the service. 



### Quick Start

Build the client docker image:
```
docker build . -f Dockerfile/Dockerfile.client -t sherpa_triton_client:latest --network host
```
We do it in the built client container, now let's start the container.

```bash
docker run -ti --net host --name sherpa_client -v $PWD/client:/ws/client sherpa_triton_client:latest
cd /ws/client
```

In the docker container, run the client script to do ASR inference.

```bash
# Test one audio using offline ASR
python3 client.py --audio_file=./test_wavs/1089-134686-0001.wav --url=localhost:8001

# Test one audio using streaming ASR
python3 client.py --audio_file=./test_wavs/1089-134686-0001.wav --url=localhost:8001 --streaming
```

The above command sends a single audio `1089-134686-0001.wav` to the server and get the result. `--url` option specifies the IP and port of the server, in our example, we set the server and client on the same machine, therefore IP is `localhost`, and we use port `8001` since it is the default port for gRPC in Triton. But if your client is not on the same machine as the server, you should change this option.

You can also test a bunch of audios together with the client. Just specify the path of `wav.scp` with `--wavscp` option, set the path of test set directory with `--data_dir` option, and set the path of ground-truth transcript file with `--trans` option, the client will infer all the audios in test set and calculate the CER upon the test set.

```bash
# Test a bunch of audios
python3 client.py --wavscp=./test_wavs/wav.scp --data_dir=./test_wavs/ --trans=./test_wavs/trans.txt

python3 client.py --wavscp=./test_wavs/wav.scp --data_dir=./test_wavs/ --trans=./test_wavs/trans.txt --streaming
```

### Performance Test

We use [perf_analyzer](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md) to test the performance of ASR service. Before this, we need to generate the test input data.

Still in the container client, run:

```bash
cd /ws/client
python3 generate_perf_input.py --audio_file=test_wavs/1089-134686-0001.wav

perf_analyzer -m conformer_transducer -b 1 -a -p 20000 --concurrency-range 100:200:50 -i gRPC --input-data=offline_input.json  -u localhost:8001
```
Similarly, for streaming ASR test:

```bash
cd /ws/client
python3 generate_perf_input.py --audio_file=test_wavs/1089-134686-0001.wav --streaming

perf_analyzer -m conformer_transducer -b 1 -a -p 20000 --concurrency-range 100:200:50 -i gRPC --input-data=online_input.json  -u localhost:8001 --streaming
```

Where:
- `-m` option indicates the name of the served model;
- `-p` option is the mearsurement window, which indicates in what time duration to collect the metrics;
- `-v` option turns on the verbose model;
- `-i` option is for choosing the networking protocol, you can choose `HTTP` or `gRPC` here;
- `-u` option sets the url of the service in the form of `<IP Adrress>:<Port>`, but notice that port `8000` corresponds to HTTP protocol while port `8001` corresponds to gRPC protocol;
- `-b` option indicates the batch size of the input requests used fo testing; since we simulate individual users sending requests, we set batch size here to `1`;
- `-a` option controls the analyzer to send requests in an asynchronous way, if this option is not applied, the requests will be sent in synchronous way;
- `--input-data` option points to the path of the json file containing the real input data
- `--concurrency-range` option is an important one, it indicates the concurrency of the requests which defines the pressure we will give to the server.
- You can also set `-f` option to set the path of testing result file;
- You can also set `--max-threads` option to set the number of threads used to send test request, it should be set to the number of CPU cores in your test machine.

As described above, if you want to send request with batch size > 1:

```bash
perf_analyzer -m attention_rescoring -b 16 -a -p20000 --concurrency-range 100:200:50 -i gRPC --input-data=./input.json  -u localhost:8001
```

## Benchmarks

In this section, we show a reference performance benchamrk for the non-streaming ASR based on Icefall. 

Notice that we send the test requests all with same length, so that Triton server will pack several requests into a batch. But in the real production cases, the lengths of all the incoming requests will not be the same, therefore, in order to allow Triton batch the requests to improve throughput, you need to try to padding the input to a specific length at the client side.

### FP32 Performance on Single GPU

First we give the performance benchmark of FP32 precision tested on single T4. We use test audios with two different lengths, 5 seconds, 8 seconds and 10 seconds. And test requests are sent with batch size 1.

#### FP32 Performance of Small Model (80Mb parameters) for Librispeech
TODO
