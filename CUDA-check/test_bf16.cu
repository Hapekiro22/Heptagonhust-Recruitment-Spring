// 创建一个简单的测试程序 test_bf16.cu
#include <cuda_bf16.h>
#include <stdio.h>

__global__ void test_bf16_kernel(__nv_bfloat16* out) {
    __nv_bfloat16 a = __float2bfloat16(1.5f);
    __nv_bfloat16 b = __float2bfloat16(2.5f);
    __nv_bfloat16 c = a + b;
    out[0] = c;
}

int main() {
    __nv_bfloat16 *d_out, *h_out;
    h_out = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, sizeof(__nv_bfloat16));
    
    test_bf16_kernel<<<1, 1>>>(d_out);
    cudaMemcpy(h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    float result = __bfloat162float(*h_out);
    printf("BF16 test: 1.5 + 2.5 = %f\n", result);
    
    // 测试转换
    float test_values[] = {-210.0f, -40.0f, 40.0f, 210.0f, 0.062f, 0.125f};
    printf("Testing BF16 conversions:\n");
    for (int i = 0; i < 6; i++) {
        __nv_bfloat16 bf_val = __float2bfloat16(test_values[i]);
        float back = __bfloat162float(bf_val);
        printf("Original: %f, Converted: %f\n", test_values[i], back);
    }
    
    cudaFree(d_out);
    free(h_out);
    return 0;
}