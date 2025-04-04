#!/bin/bash
#SBATCH -n 1
#SBATCH -o cuda_tests-%j.out
#SBATCH -e cuda_tests-%j.err
#SBATCH -c 4
#SBATCH --gres=gpu:1

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
eval $(spack load --sh cuda@12.8.0)


./test-bf16gemm