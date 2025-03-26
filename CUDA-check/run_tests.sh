#!/bin/bash
#SBATCH -n 1
#SBATCH -o cuda_tests-%j.out
#SBATCH -e cuda_tests-%j.err
#SBATCH -c 4
#SBATCH --gres=gpu:1

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

echo "=============== 环境信息 ==============="
hostname
date
echo "CUDA 库路径: $LD_LIBRARY_PATH"
echo ""

echo "=============== 基本 CUDA 功能测试 ==============="
./test_cuda_basic
echo ""

echo "=============== cuBLAS 矩阵乘法测试 ==============="
./test_cublas
echo ""

echo "=============== 特定 sgemm_cublas 函数测试 ==============="
./test_your_sgemm
echo ""

echo "所有测试完成！"
