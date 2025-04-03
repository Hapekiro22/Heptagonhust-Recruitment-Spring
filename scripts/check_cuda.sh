#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o cuda_info-hepnode2.out
#SBATCH -e cuda_info.err
#SBATCH --exclude hepnode0,hepnode2,hepnode1

echo "======================= CUDA 信息查询 ======================="
echo "主机名: $(hostname)"
echo "执行日期: $(date)"
echo "======================= 环境变量 ======================="
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo "======================= CUDA 安装位置 ======================="
echo "CUDA软链接指向: $(ls -la /usr/local/cuda 2>/dev/null || echo 'CUDA 软链接不存在')"
echo "CUDA可执行文件位置: $(which nvcc 2>/dev/null || echo 'nvcc not found')"
echo "CUDA版本: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"

echo "======================= CUDA 库文件 ======================="
echo "CUDA运行时库: $(ldconfig -p | grep libcudart)"
echo "CUBLAS库: $(ldconfig -p | grep libcublas)"

echo "======================= GPU 设备信息 ======================="
nvidia-smi || echo "nvidia-smi 命令不可用"

echo "======================= CUDA 头文件位置 ======================="
find /usr/include -name cuda.h 2>/dev/null
find /usr/local -name cuda.h 2>/dev/null

echo "======================= 可能的 CUDA 路径 ======================="
find /usr -name cuda -type d 2>/dev/null | grep -v "incomplete\|samples"
find /opt -name cuda -type d 2>/dev/null 2>/dev/null

echo "======================= 模块系统检查 ======================="
module avail 2>/dev/null | grep -i cuda || echo "模块系统不可用或未找到 CUDA 模块"

echo "======================= 检查 cuBLAS 可用性 ======================="
find /usr -name "*cublas*.h" 2>/dev/null
find /usr/local -name "*cublas*.h" 2>/dev/null