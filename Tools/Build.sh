#!/bin/bash

# Goto dir
cd ..

# Ensure the Build directory exists
mkdir -p Build

# Change to the Build directory
cd Build

# Run CMake to generate build files
cmake ..

# Build the project
make

# Move the binary to the Binaries directory
mv opencl_gpu_detection ../Binaries/