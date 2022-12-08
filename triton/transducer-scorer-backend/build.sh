#!/usr/bin/bash

mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
