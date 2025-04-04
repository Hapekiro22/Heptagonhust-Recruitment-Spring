// precision_kernels.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {
    // FP32到BF16的转换内核
    __global__ void convertFP32ToBF16Kernel(float* input, __nv_bfloat16* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = __float2bfloat16(input[idx]);
        }
    }
    
    // 用于从C++代码调用的封装函数
    cudaError_t launchConvertFP32ToBF16(float* input, __nv_bfloat16* output, int size, cudaStream_t stream) {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        convertFP32ToBF16Kernel<<<numBlocks, blockSize, 0, stream>>>(input, output, size);
        return cudaGetLastError();
    }
}