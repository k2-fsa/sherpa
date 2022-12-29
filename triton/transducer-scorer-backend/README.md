### Custom backend for scorer module

Currently, only support model_repo_offline

```
# In server docker container,
apt-get install rapidjson-dev
pip3 install cmake==3.22
rm /usr/bin/cmake
ln /usr/local/bin/cmake /usr/bin/cmake
cmake --version
bash build.sh

# Put the generated libtriton_scorer.so under model_repo_offline/scorer/2
# Also change backend name in model_repo_offline/scorer/config.pbtxt from backend:"python" to backend: "scorer"

```
