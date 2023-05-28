# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


rm -f /usr/lib/x86_64-linux-gnu/libnvinfer*
rm -f /usr/lib/x86_64-linux-gnu/libnvcaffe_parser.so*
rm -f /usr/lib/x86_64-linux-gnu/libnvonnxparser.so*
rm -f /usr/lib/x86_64-linux-gnu/libnvparsers.so*
rm -f /opt/tensorrt/bin/trtexec
 
cp -rf lib/* /usr/lib/x86_64-linux-gnu/
cp -f ./targets/x86_64-linux-gnu/bin/trtexec /opt/tensorrt/bin/

cd python
python3 -m pip install tensorrt-8.5.3.1-cp38-none-linux_x86_64.whl

