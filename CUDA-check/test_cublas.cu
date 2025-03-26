#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void check_cublas_error(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS 错误 (%s): ", msg);
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED: printf("CUBLAS_STATUS_NOT_INITIALIZED\n"); break;
            case CUBLAS_STATUS_ALLOC_FAILED: printf("CUBLAS_STATUS_ALLOC_FAILED\n"); break;
            case CUBLAS_STATUS_INVALID_VALUE: printf("CUBLAS_STATUS_INVALID_VALUE\n"); break;
            case CUBLAS_STATUS_ARCH_MISMATCH: printf("CUBLAS_STATUS_ARCH_MISMATCH\n"); break;
            case CUBLAS_STATUS_MAPPING_ERROR: printf("CUBLAS_STATUS_MAPPING_ERROR\n"); break;
            case CUBLAS_STATUS_EXECUTION_FAILED: printf("CUBLAS_STATUS_EXECUTION_FAILED\n"); break;
            case CUBLAS_STATUS_INTERNAL_ERROR: printf("CUBLAS_STATUS_INTERNAL_ERROR\n"); break;
            case CUBLAS_STATUS_NOT_SUPPORTED: printf("CUBLAS_STATUS_NOT_SUPPORTED\n"); break;
            case CUBLAS_STATUS_LICENSE_ERROR: printf("CUBLAS_STATUS_LICENSE_ERROR\n"); break;
            default: printf("未知错误码: %d\n", status);
        }
        exit(1);
    }
}

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA 错误 (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void print_matrix(const char* name, float* matrix, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 6; i++) {
        for (int j = 0; j < cols && j < 6; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("%s\n", cols > 6 ? "..." : "");
    }
    if (rows > 6) printf("...\n");
    printf("\n");
}

int main() {
    printf("========= cuBLAS 矩阵乘法测试 =========\n");
    
    // 初始化 CUDA
    cudaError_t err = cudaSetDevice(0);
    check_cuda_error(err, "设置设备");
    
    // 查询设备信息
    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    check_cuda_error(err, "查询内存信息");
    
    printf("CUDA 内存: 总计 %.2f GB, 可用 %.2f GB\n", 
           total_mem / (1024.0 * 1024.0 * 1024.0),
           free_mem / (1024.0 * 1024.0 * 1024.0));
    
    // 初始化 cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    check_cublas_error(status, "创建 cuBLAS 句柄");
    
    // 测试不同大小的矩阵
    int sizes[] = {32, 256, 1024, 4096};
    for (int s = 0; s < sizeof(sizes)/sizeof(int); s++) {
        int N = sizes[s];
        printf("\n测试矩阵大小: %d x %d\n", N, N);
        
        // 矩阵 A: NxN 全 1
        // 矩阵 B: NxN 全 1
        // 矩阵 C = A * B: NxN (每个元素应为 N)
        
        size_t matrix_bytes = N * N * sizeof(float);
        size_t total_bytes = matrix_bytes * 3;
        
        if (total_bytes > free_mem * 0.9) {
            printf("内存不足：需要 %.2f GB, 跳过此测试\n", 
                   total_bytes / (1024.0 * 1024.0 * 1024.0));
            continue;
        }
        
        // 分配主机内存
        float *h_A = (float*)malloc(matrix_bytes);
        float *h_B = (float*)malloc(matrix_bytes);
        float *h_C = (float*)malloc(matrix_bytes);
        
        if (!h_A || !h_B || !h_C) {
            printf("主机内存分配失败\n");
            exit(1);
        }
        
        // 初始化矩阵
        for (int i = 0; i < N * N; i++) {
            h_A[i] = 1.0f;
            h_B[i] = 1.0f;
        }
        
        // 调试打印小矩阵
        if (N <= 32) {
            print_matrix("矩阵 A", h_A, N, N);
            print_matrix("矩阵 B", h_B, N, N);
        }
        
        // 分配设备内存
        float *d_A, *d_B, *d_C;
        err = cudaMalloc(&d_A, matrix_bytes);
        check_cuda_error(err, "分配设备内存 A");
        
        err = cudaMalloc(&d_B, matrix_bytes);
        check_cuda_error(err, "分配设备内存 B");
        
        err = cudaMalloc(&d_C, matrix_bytes);
        check_cuda_error(err, "分配设备内存 C");
        
        // 复制数据到设备
        err = cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice);
        check_cuda_error(err, "复制 A 到设备");
        
        err = cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice);
        check_cuda_error(err, "复制 B 到设备");
        
        // 执行矩阵乘法
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // C = A * B
        printf("执行 cublasSgemm (N=%d)...\n", N);
        status = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,  // A 是 NxN
            d_B, N,  // B 是 NxN
            &beta,
            d_C, N   // C 是 NxN
        );
        check_cublas_error(status, "执行 SGEMM");
        
        // 同步确保完成
        err = cudaDeviceSynchronize();
        check_cuda_error(err, "设备同步");
        
        // 复制结果回主机
        err = cudaMemcpy(h_C, d_C, matrix_bytes, cudaMemcpyDeviceToHost);
        check_cuda_error(err, "复制结果到主机");
        
        // 验证结果
        if (N <= 32) {
            print_matrix("结果矩阵 C", h_C, N, N);
        }
        
        // 检查结果
        bool correct = true;
        for (int i = 0; i < N * N && correct; i++) {
            if (fabs(h_C[i] - N) > 0.001f) {
                printf("验证失败: C[%d] = %f (应为 %d)\n", i, h_C[i], N);
                correct = false;
            }
        }
        
        if (correct) {
            printf("矩阵乘法验证通过 (N=%d)！\n", N);
        }
        
        // 释放内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
    }
    
    cublasDestroy(handle);
    printf("\ncuBLAS 测试完成!\n");
    
    return 0;
}
