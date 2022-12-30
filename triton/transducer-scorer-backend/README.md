### Custom backend for scorer module

Currently, only support transducer greedy search method for model_repo_offline

```
# In server docker container,
apt-get install rapidjson-dev
pip3 install cmake==3.22
rm /usr/bin/cmake
ln /usr/local/bin/cmake /usr/bin/cmake
cmake --version

# To avoid torch ABI issue, download libtorch here. 
wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip 
unzip -d $(pwd)/ libtorch-cxx11-abi-shared-with-deps-1.13.1+cu116.zip
export Torch_DIR=$(pwd)/libtorch
bash build.sh

# Put the generated libtriton_scorer.so under model_repo_offline/scorer/2
# Also change backend name in model_repo_offline/scorer/config.pbtxt from backend:"python" to backend: "scorer"

```
