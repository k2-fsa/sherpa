#!/usr/bin/env bash

set -ex

./generate_download.py
./generate_build_cpu.py

ls -lh ./generated/*
