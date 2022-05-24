## Introduction

An ASR server framework in **Python**, aiming to support both streaming
and non-streaming recognition.

**Note**: Only non-streaming recognition is implemented at present. We
will add streaming recognition later.

CPU-bound tasks, such as neural network computation, are implemented in
C++; while IO-bound tasks, such as socket communication, are implemented
in Python.

**Caution**: We assume the model is trained using pruned stateless RNN-T
from [icefall][icefall] and it is from a directory like
`pruned_transducer_statelessX` where `X` >=2.

## Installation

First, you have to install `PyTorch` and `torchaudio`. PyTorch 1.10 is known
to work. Other versions may also work.

Second, clone this repository

```bash
git clone https://github.com/k2-fsa/sherpa
cd sherpa
pip install -r ./requirements.txt
```

Third, install the C++ extension of `sherpa`. You can use one of
the following methods.

### Option 1: Use `pip`

```bash
pip install --verbose k2-sherpa
```

### Option 2: Build from source with `setup.py`

```bash
python3 setup.py install
```

### Option 3: Build from source with `cmake`

```bash
mkdir build
cd build
cmake ..
make -j
export PYTHONPATH=$PWD/../sherpa/python:$PWD/lib:$PYTHONPATH
```


## Usage

First, check that `sherpa` has been installed successfully:

```bash
python3 -c "import sherpa; print(sherpa.__version__)"
```

It should print the version of `sherpa`.

### Start the server

To start the server, you need to first generate two files:

- (1) The torch script model file. You can use `export.py --jit=1` in
`pruned_transducer_statelessX` from [icefall][icefall].

- (2) The BPE model file. You can find it in `data/lang_bpe_XXX/bpe.model`
in [icefall][icefall], where `XXX` is the number of BPE tokens used in
the training.

With the above two files ready, you can start the server with the
following command:

```bash
sherpa/bin/offline_server.py \
  --port 6006 \
  --num-device 0 \
  --max-batch-size 10 \
  --max-wait-ms 5 \
  --feature-extractor-pool-size 5 \
  --nn-pool-size 1 \
  --nn-model-filename ./path/to/exp/cpu_jit.pt \
  --bpe-model-filename ./path/to/data/lang_bpe_500/bpe.model &
```

You can use `./sherpa/bin/offline_server.py --help` to view the help message.

We provide a pretrained model using the LibriSpeech dataset at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>

The following shows how to use the above pretrained model to start the server.

```bash
git lfs install
git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

sherpa/bin/offline_server.py \
  --port 6006 \
  --num-device 0 \
  --max-batch-size 10 \
  --max-wait-ms 5 \
  --feature-extractor-pool-size 5 \
  --nn-pool-size 1 \
  --nn-model-filename ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp/cpu_jit.pt \
  --bpe-model-filename ./icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/data/lang_bpe_500/bpe.model
```

### Start the client
After starting the server, you can use the following command to start the client:

```bash
./sherpa/bin/offline_client.py \
    --server-addr localhost \
    --server-port 6006 \
    /path/to/foo.wav \
    /path/to/bar.wav
```

You can use `./sherpa/bin/offline_client.py --help` to view the usage message.

The following shows how to use the client to send some test waves to the server
for recognition.

```bash
sherpa/bin/offline_client.py \
  --server-addr localhost \
  --server-port 6006 \
  icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13//test_wavs/1089-134686-0001.wav \
  icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13//test_wavs/1221-135766-0001.wav \
  icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13//test_wavs/1221-135766-0002.wav
```

### RTF test

We provide a demo [./sherpa/bin/decode_mainifest.py](./sherpa/bin/decode_mainifest.py)
to decode the `test-clean` dataset from the LibriSpeech corpus.

It creates 50 connections to the server using websockets and sends audio files
to the server for recognition.

At the end, it will display the RTF and the WER.

[icefall]: https://github.com/k2-fsa/icefall/
