#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel(float *d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[idx] = idx;
}

int main() {
    printf("========= CUDA 基础功能测试 =========\n");
    
    // 查询 CUDA 设备
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("检测到 %d 个 CUDA 设备\n", deviceCount);
    if (deviceCount == 0) {
        printf("未找到可用的 CUDA 设备\n");
        return 1;
    }
    
    // 打印设备信息
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("设备 %d: %s\n", i, prop.name);
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  多处理器数量: %d\n", prop.multiProcessorCount);
        printf("  时钟频率: %.0f MHz\n", prop.clockRate / 1000.0);
        printf("  显存总线宽度: %d bits\n", prop.memoryBusWidth);
    }
    
    // 选择第一个设备
    cudaSetDevice(0);
    
    // 分配和使用内存
    const int SIZE = 256;
    float h_out[SIZE];
    float *d_out = NULL;
    
    error = cudaMalloc(&d_out, SIZE * sizeof(float));
    if (error != cudaSuccess) {
        printf("cudaMalloc 失败: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // 运行内核
    hello_kernel<<<1, SIZE>>>(d_out);
    
    // 检查内核执行错误
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 内核错误: %s\n", cudaGetErrorString(error));
        cudaFree(d_out);
        return 1;
    }
    
    // 等待内核完成
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("CUDA 同步错误: %s\n", cudaGetErrorString(error));
        cudaFree(d_out);
        return 1;
    }
    
    // 复制回结果
    error = cudaMemcpy(h_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("cudaMemcpy 失败: %s\n", cudaGetErrorString(error));
        cudaFree(d_out);
        return 1;
    }
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_out[i] != i) {
            printf("验证错误: h_out[%d] = %f (应为 %d)\n", i, h_out[i], i);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("基础 CUDA 功能测试通过!\n");
    }
    
    cudaFree(d_out);
    return 0;
}
