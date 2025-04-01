#!/bin/bash

#SBATCH -n 1
#SBATCH -o nsys_results/out/nsys-job-%j.out
#SBATCH -e nsys_results/err/nsys-job-%j.err
#SBATCH -c 64
#SBATCH --exclusive
#SBATCH --exclude hepnode0,hepnode2,hepnode3
#SBATCH --gres=gpu:1

# 参数检查
if [ -z "$1" ]; then
    echo "错误: 缺少版本参数!"
    echo "用法: sbatch run_nsight.sh <版本号> [配置类型]"
    echo "示例: sbatch run_nsight.sh v0.5.0 1"
    exit 1
fi

# 设置变量
directory=nsys_results
version=$1
config_type=${2:-1}  # 默认使用 vgg16 配置 (1)

# 确保输出目录存在
mkdir -p $directory/out $directory/err $directory/${version}

# 设置 CUDA 环境变量
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# 加载必要的模块
eval $(spack load --sh nvidia-nsight-systems@2024.6.1)
eval $(spack load --sh cuda@12.8.0)

# 打印环境信息
echo "====== 环境信息 ======"
echo "主机名: $(hostname)"
echo "日期时间: $(date)"
echo "CUDA版本:"
nvcc --version
echo "Nsight Systems版本:"
nsys --version
echo "GPU信息:"
nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
echo "======================"

# 确定配置文件
if [ "$config_type" == "1" ]; then
    CONFIG_FILE="conf/vgg16.conf"
    CONFIG_NAME="vgg16"
else
    CONFIG_FILE="conf/small.conf" 
    CONFIG_NAME="small"
fi

echo "使用配置文件: $CONFIG_FILE"
echo "开始分析: $(date)"

# 输出文件名
OUTPUT_BASE="${directory}/${version}/${version}_${CONFIG_NAME}"
PROFILE_OUTPUT="${OUTPUT_BASE}"
RESULT_FILE="${OUTPUT_BASE}.out"

# 执行性能分析
nsys profile \
  --trace=cuda,cublas,osrt,nvtx \
  --cuda-memory-usage=true \
  --stats=true \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --export=sqlite,json \
  --force-overwrite=true \
  -o "${PROFILE_OUTPUT}" \
  numactl --cpunodebind=0-3 --membind=0-3 ./winograd $CONFIG_FILE > "${RESULT_FILE}" 2>&1

# 检查 nsys 命令执行状态
if [ $? -ne 0 ]; then
    echo "错误: Nsight Systems 分析失败!"
    exit 1
fi

echo "主分析完成: $(date)"
echo "结果保存在: ${PROFILE_OUTPUT}.nsys-rep"

# 生成详细报告
echo "生成详细报告..."

REPORTS_DIR="${directory}/${version}/${version}_${CONFIG_NAME}"
mkdir -p $REPORTS_DIR

nsys stats "${PROFILE_OUTPUT}.nsys-rep" --report gputrace > "${REPORTS_DIR}/gputrace.txt"
nsys stats "${PROFILE_OUTPUT}.nsys-rep" --report cudaapisum > "${REPORTS_DIR}/cudaapisum.txt"
nsys stats "${PROFILE_OUTPUT}.nsys-rep" --report cudaapitrace > "${REPORTS_DIR}/cudaapitrace.txt"
nsys stats "${PROFILE_OUTPUT}.nsys-rep" --report gpukernsum > "${REPORTS_DIR}/gpukernsum.txt"
nsys stats "${PROFILE_OUTPUT}.nsys-rep" --report gpumemtimeline > "${REPORTS_DIR}/gpumemtimeline.txt"
nsys stats "${PROFILE_OUTPUT}.nsys-rep" --report gpumemsizesum > "${REPORTS_DIR}/gpumemsizesum.txt"

# 执行额外的CUDA核心分析(如果需要)
if [ "$config_type" == "1" ]; then
    echo "执行额外的批处理矩阵乘法性能分析..."
    
    # 可选：使用 ncu 进行特定内核的详细分析
    # ncu --metrics sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second --kernel-regex "cublasSgemm|sgemm" ./winograd $CONFIG_FILE > "${REPORTS_DIR}/kernel_metrics.txt" 2>&1
fi

echo "所有分析完成: $(date)"
echo "报告保存在: ${REPORTS_DIR}/"

# 创建简要摘要
echo "===== 性能分析摘要 =====" > "${OUTPUT_BASE}_summary.txt"
echo "分析版本: $version" >> "${OUTPUT_BASE}_summary.txt"
echo "配置: $CONFIG_NAME" >> "${OUTPUT_BASE}_summary.txt" 
echo "日期: $(date)" >> "${OUTPUT_BASE}_summary.txt"
echo "" >> "${OUTPUT_BASE}_summary.txt"
echo "CUDA API 调用总计:" >> "${OUTPUT_BASE}_summary.txt"
grep -A 10 "CUDA API Statistics:" "${REPORTS_DIR}/cudaapisum.txt" >> "${OUTPUT_BASE}_summary.txt"
echo "" >> "${OUTPUT_BASE}_summary.txt"
echo "GPU 内核执行总计:" >> "${OUTPUT_BASE}_summary.txt"
grep -A 10 "CUDA Kernel Statistics:" "${REPORTS_DIR}/gpukernsum.txt" >> "${OUTPUT_BASE}_summary.txt"

echo "摘要已保存到: ${OUTPUT_BASE}_summary.txt"