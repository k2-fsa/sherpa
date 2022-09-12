#!/usr/bin/env bash
#
# Copyright      2021  Xiaomi Corp.       (author: Fangjun Kuang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following environment variables are supposed to be set by users
#
# - SHERPA_TORCH_VERSION
#     The PyTorch version. Example:
#
#       export SHERPA_TORCH_VERSION=1.7.1
#
#     Defaults to 1.7.1 if not set.
#
# - SHERPA_CONDA_TOKEN
#     If not set, auto upload to anaconda.org is disabled.
#
#     Its value is from https://anaconda.org/k2-fsa-sherpa/settings/access
#      (You need to login as user k2-fsa-sherpa to see its value)
#
set -e
export CONDA_BUILD=1

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
sherpa_dir=$(cd $cur_dir/.. && pwd)

cd $sherpa_dir

export SHERPA_ROOT_DIR=$sherpa_dir
echo "SHERPA_ROOT_DIR: $SHERPA_ROOT_DIR"

SHERPA_PYTHON_VERSION=$(python -c "import sys; print('.'.join(sys.version.split('.')[:2]))")

if [ -z $SHERPA_TORCH_VERSION ]; then
  echo "env var SHERPA_TORCH_VERSION is not set, defaults to 1.7.1"
  SHERPA_TORCH_VERSION=1.7.1
fi

# Example value: 3.8
export SHERPA_PYTHON_VERSION

# Example value: 1.7.1
export SHERPA_TORCH_VERSION

# conda remove -q pytorch
# conda clean -q -a

if [ -z $SHERPA_CONDA_TOKEN ]; then
  echo "Auto upload to anaconda.org is disabled since SHERPA_CONDA_TOKEN is not set"
  conda build --no-test --no-anaconda-upload -c pytorch -c k2-fsa -c kaldifeat -c kaldi_native_io ./scripts/conda-cpu/sherpa
else
  conda build --no-test -c pytorch -c k2-fsa -c kaldifeat -c kaldi_native_io --token $SHERPA_CONDA_TOKEN ./scripts/conda-cpu/sherpa
fi
