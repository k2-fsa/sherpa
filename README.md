<div align="center">
<img src="./docs/source/python/installation/pic/os-brightgreen.svg">
<img src="./docs/source/python/installation/pic/python_ge_3.7-blue.svg">
<img src="./docs/source/python/installation/pic/pytorch_ge_1.6.0-blueviolet.svg">
<img src="./docs/source/python/installation/pic/cuda_ge_10.1-orange.svg">
</div>

[![Documentation Status](https://github.com/k2-fsa/sherpa/actions/workflows/build-doc.yml/badge.svg)](https://k2-fsa.github.io/sherpa/)

Try `sherpa` from within your browser without installing anything:
<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>

See <https://k2-fsa.github.io/sherpa/python/huggingface/> for more details.



# sherpa

`sherpa` is an open-source speech-to-text (i.e., speech recognition) framework,
focusing **exclusively** on end-to-end (E2E) models, namely transducer- and
CTC-based models.

**Note**: There is no plan to support attention-based encoder-decoder (AED)
models.

## Installation

Please first install:

  - [PyTorch](https://pytorch.org/get-started/locally/)
  - [k2][k2]
  - [kaldifeat][kaldifeat]

```bash
git clone https://github.com/k2-fsa/sherpa
cd sherpa
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/software/sherpa
  ..
make install # or make install/strip

export PATH=$HOME/software/sherpa/bin:$PATH
```

or

```bash
git clone https://github.com/k2-fsa/sherpa
cd sherpa

python3 setup.py bdist_wheel
pip install ./dist/k2_sherpa-*.whl

# Please don't use `python3 setup.py install`.
# Otherwise, you won't have access to pre-compiled binaries
```

See **documentation** at <https://k2-fsa.github.io/sherpa/> for more usages.

[k2]: http://github.com/k2-fsa/k2
[kaldifeat]: https://github.com/csukuangfj/kaldifeat
