#!/usr/bin/env bash

echo "PREFIX: $PREFIX"

# conda install -y -q -c pytorch pytorch={{ environ.get('SHERPA_TORCH_VERSION') }} cpuonly

os=$(uname -s)

if [ $os != "Linux" ]; then
  export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
  ls -lh $PREFIX/lib/libmkl*
  cp -v $PREFIX/lib/libmkl_intel_ilp64.dylib $PREFIX/lib/libmkl_intel_ilp64.2.dylib # [osx]
  cp -v $PREFIX/lib/libmkl_core.dylib $PREFIX/lib/libmkl_core.2.dylib # [osx]
  cp -v $PREFIX/lib/libmkl_intel_thread.dylib $PREFIX/lib/libmkl_intel_thread.2.dylib # [osx]
  ls -lh $PREFIX/lib/libmkl* # [not win]
fi

python setup.py install --single-version-externally-managed --record=record.txt

if [ $os == "Linux" ]; then
  cp build/lib.linux-x86_64-*/sherpa/bin/sherpa $PREFIX/bin   # [linux]
else
  cp build/lib.macosx-*-x86_64-*/sherpa/bin/sherpa $PREFIX/bin # [osx]
fi
chmod +x $PREFIX/bin/sherpa
