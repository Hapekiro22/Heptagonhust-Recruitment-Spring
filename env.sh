#!/bin/bash

# Load Intel Vtune
echo "Load Intel Vtune"
eval $(spack load --sh intel-oneapi-vtune)
vtune --version


# Load CUDA 12.8.0
echo "Load CUDA 12.8.0"
eval $(spack load --sh cuda@12.8.0)
nvcc --version