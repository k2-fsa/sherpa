## Introduction

An ASR server framework supporting both streaming and non-streaming recognition.

Most parts will be implemented in Python, while CPU-bound tasks are implemented
in C++, which are called by Python threads with the GIL being released.

## TODOs

- [ ] Support non-streaming recognition
- [ ] Documentation for installation and usage
- [ ] Support streaming recognition


## Usage:

First, you have to install PyTorch. PyTorch == 1.10 is known to work.
Other versions may also work.

```bash
pip install websockets
git clone  https://github.com/k2-fsa/sherpa.git
cd sherpa
mkdir build
cd build
cmake ..
make -j
export PYTHONPATH=$PWD/../sherpa/python:$PWD/lib:$PYTHONPATH
cd ../sherpa/bin

./offline_server.py

# Open a new terminal, and run
./offline_client.py

# or run the following script to decode test-clean of librispeech
./decode_mainifest.py
```

We will make it installable by using `pip install`, `conda instal`, or `python3 setup.py install`.
