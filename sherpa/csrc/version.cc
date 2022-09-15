/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sherpa/csrc/version.h"

#include <iostream>
#include <sstream>
int main() {
  std::ostringstream os;
  os << "sherpa version: " << sherpa::kVersion << "\n";
  os << "build type: " << sherpa::kBuildType << "\n";
  os << "OS used to build sherpa: " << sherpa::kOS << "\n";
  os << "sherpa git sha1: " << sherpa::kGitSha1 << "\n";
  os << "sherpa git date: " << sherpa::kGitDate << "\n";
  os << "sherpa git branch: " << sherpa::kGitBranch << "\n";

  os << "PyTorch version used to build sherpa: " << sherpa::kTorchVersion
     << "\n";
  os << "CUDA version: " << sherpa::kCudaVersion << "\n";
  os << "cuDNN version: " << sherpa::kCudnnVersion << "\n";

  os << "k2 version used to build sherpa: " << sherpa::kK2Version << "\n";
  os << "k2 git sha1: " << sherpa::kK2GitSha1 << "\n";
  os << "k2 git date: " << sherpa::kK2GitDate << "\n";
  os << "k2 with cuda: " << sherpa::kK2WithCuda << "\n";

  os << "kaldifeat version used to build sherpa: " << sherpa::kKaldifeatVersion
     << "\n";

  os << "cmake version: " << sherpa::kCMakeVersion << "\n";
  os << "compiler ID: " << sherpa::kCompilerID << "\n";
  os << "compiler: " << sherpa::kCompiler << "\n";
  os << "compiler version: " << sherpa::kCompilerVersion << "\n";
  os << "cmake CXX flags: " << sherpa::kCMakeCxxFlags << "\n";
  os << "Python version: " << sherpa::kPythonVersion << "\n";

  std::cerr << os.str();

  return 0;
}
