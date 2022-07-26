#!/usr/bin/env bash

# This script installs kaldifeat into the directory ~/tmp/icefall
# which is cached by GitHub actions for later runs.

mkdir -p ~/tmp
cd ~/tmp
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
