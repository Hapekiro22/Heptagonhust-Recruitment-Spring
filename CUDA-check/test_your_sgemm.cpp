#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 定义 cublas 句柄
static cublasHandle_t cublas_handle = nullptr;

// CUDA_ERROR CHECK
#define CHECK_CUDA_ERROR(call) { \
  cudaError_t err = call;      \
  if (err != cudaSuccess) {    \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      return; \
  } \
}

// cuBLAS_ERROR CHECK
#define CHECK_CUBLAS_ERROR(call) { \
  cublasStatus_t status = call;  \
  if (status != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS error at %s:%d - code: %d\n", __FILE__, __LINE__, status); \
      return; \
  } \
}

// 模拟 init_cublas 函数
bool init_cublas() {
  printf("初始化 cuBLAS...\n");
  if(cublas_handle == nullptr) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
      fprintf(stderr, "无法获取 CUDA 设备数量: %s\n", cudaGetErrorString(err));
      return false;
    }
    
    if(device_count == 0) {
      fprintf(stderr, "未找到 CUDA 设备\n");
      return false;
    }
    printf("找到 %d 个 CUDA 设备\n", device_count);
    
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      fprintf(stderr, "无法设置 CUDA 设备: %s\n", cudaGetErrorString(err));
      return false;
    }
    printf("成功设置 CUDA 设备 0\n");
    
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "无法创建 cuBLAS 句柄: %d\n", status);
      return false;
    }
    printf("成功创建 cuBLAS 句柄\n");
  }

  return true;
}

// 与你的代码类似的 sgemm_cublas 函数
void sgemm_cublas(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) 
{
    printf("sgemm_cublas: M=%ld, N=%ld, K=%ld\n", M, N, K);
    
    // 初始化 cuBLAS
    if(!init_cublas()) {
        fprintf(stderr, "cuBLAS 初始化失败\n");
        return;
    }
    
    // 检查内存
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        fprintf(stderr, "无法查询 GPU 内存: %s\n", cudaGetErrorString(err));
        return;
    }
    
    size_t required_mem = sizeof(float) * (M * K + N * K + M * N);
    printf("内存要求: %.2f MB, 可用: %.2f MB\n", 
           required_mem / (1024.0 * 1024.0), 
           free_mem / (1024.0 * 1024.0));
    
    if (required_mem > free_mem * 0.9) {
        printf("内存不足, 跳过 GPU 计算\n");
        return;
    }
    
    // 分配 GPU 内存
    printf("正在分配 GPU 内存...\n");
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    try {
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, sizeof(float) * M * K));
        printf("d_A 分配成功\n");
        
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, sizeof(float) * N * K));
        printf("d_B 分配成功\n");
        
        CHECK_CUDA_ERROR(cudaMalloc(&d_C, sizeof(float) * M * N));
        printf("d_C 分配成功\n");
        
        // 将数据从主机复制到设备
        printf("正在复制数据到 GPU...\n");
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, sizeof(float) * N * K, cudaMemcpyHostToDevice));
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // 使用下面原始的参数调用
        printf("正在执行 cublasSgemm (原始参数)...\n");
        // 修改为完全匹配 CPU 版本的调用
        CHECK_CUBLAS_ERROR(cublasSgemm(
            cublas_handle,       
            CUBLAS_OP_T,       
            CUBLAS_OP_N,       
            M,                 // C 的行数
            N,                 // C 的列数
            K,                 // 共同维度
            &alpha,
            d_A,               
            M,                 
            d_B,               
            K,                 
            &beta,
            d_C,               
            M
        ));
        
        printf("cublasSgemm 执行成功，正在复制结果回主机...\n");
        CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, sizeof(float) * N * M, cudaMemcpyDeviceToHost));
        
        printf("复制完成，正在释放 GPU 内存...\n");
        CHECK_CUDA_ERROR(cudaFree(d_A));
        CHECK_CUDA_ERROR(cudaFree(d_B));
        CHECK_CUDA_ERROR(cudaFree(d_C));
        printf("GPU 内存释放成功\n");
    }
    catch (...) {
        printf("执行过程中捕获到异常\n");
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
    }
}
// 符合标准矩阵乘法定义的 CPU 实现
void sgemm_cpu(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
    typedef float(*A_tensor_t)[K];
    typedef float(*B_tensor_t)[K];
    typedef float(*C_tensor_t)[M];
    A_tensor_t A_tensor = (A_tensor_t)A;
    B_tensor_t B_tensor = (B_tensor_t)B;
    C_tensor_t C_tensor = (C_tensor_t)C;
  
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        C_tensor[n][m] = 0;
        for (int64_t k = 0; k < K; ++k) {
          C_tensor[n][m] += A_tensor[m][k] * B_tensor[n][k];
        }
      }
    }
  }

void destroy_cublas() {
    if(cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

void print_matrix(const char* name, float *mat, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 6; i++) {
        for (int j = 0; j < cols && j < 6; j++) {
            printf("%8.2f ", mat[i * cols + j]);
        }
        printf(cols > 6 ? "...\n" : "\n");
    }
    if (rows > 6) printf("...\n");
}

void print_matrix_sequence(const char* name, float *mat, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows*cols;i++) {
        printf("%8.2f ", mat[i]);
        if ((i + 1) % cols == 0) {
            printf("\n");
        }
    }
}

void verify_results(float *C_gpu, float *C_cpu, int64_t M, int64_t N) {
    int errors = 0;
    for (int64_t i = 0; i < M * N; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > 1e-3) {
            if (errors < 10) {
                printf("不匹配: C_gpu[%ld] = %f, C_cpu[%ld] = %f\n", 
                       i, C_gpu[i], i, C_cpu[i]);
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("验证成功: GPU 和 CPU 结果一致!\n");
    } else {
        printf("验证失败: 发现 %d 个不匹配的元素 (总共 %ld 个)\n", errors, M * N);
    }
}

int main() {
    printf("========= 测试你的 sgemm_cublas 函数 =========\n\n");
    
    // 测试不同大小的矩阵
    int sizes[][3] = {
        {4, 2, 1},    // 小矩阵
        {64, 64, 64},    // 中等矩阵
        {128, 128, 128}, // 较大矩阵
        {512, 256, 256}, // 更大矩阵
        {512, 512, 512}  // 如果内存允许
    };
    
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int t = 0; t < num_tests; t++) {
        int64_t M = sizes[t][0];
        int64_t N = sizes[t][1];
        int64_t K = sizes[t][2];
        
        printf("\n===== 测试 %d: M=%ld, N=%ld, K=%ld =====\n", t+1, M, N, K);
        
        // 分配内存
        float *A = (float*)malloc(sizeof(float) * M * K);
        float *B = (float*)malloc(sizeof(float) * N * K);
        float *C_gpu = (float*)malloc(sizeof(float) * M * N);
        float *C_cpu = (float*)malloc(sizeof(float) * M * N);
        
        if (!A || !B || !C_gpu || !C_cpu) {
            printf("内存分配失败\n");
            return 1;
        }
        
        // 随机初始化矩阵
        for (int64_t i = 0; i < M * K; i++) {
            A[i] = (float)(rand() % 100) / 100.0f;
        }
        for (int64_t i = 0; i < N * K; i++) {
            B[i] = (float)(rand() % 100) / 100.0f;
        }
        
        // 显示矩阵（仅对于小矩阵）
        if (M <= 16 && N <= 16 && K <= 16) {
            print_matrix("矩阵 A", A, M, K);
            print_matrix("矩阵 B", B, N, K);
        }
        
        // 运行 GPU 版本
        printf("执行 GPU 矩阵乘法...\n");
        sgemm_cublas(M, N, K, A, B, C_gpu);
        
        // 运行 CPU 参考版本
        printf("执行 CPU 矩阵乘法...\n");
        sgemm_cpu(M, N, K, A, B, C_cpu);

        //转置C_gpu
        /*float *C_gpu_transposed = (float*)malloc(sizeof(float) * M * N);
    
        // 转置 GPU 结果
        for (int64_t m = 0; m < M; m++) {
            for (int64_t n = 0; n < N; n++) {
                C_gpu_transposed[n * M + m] = C_gpu[m * N + n];
            }
        }*/
        
        // 显示结果（仅对于小矩阵）
        if (M <= 16 && N <= 16) {
            print_matrix_sequence("GPU 结果", C_gpu, N, M);
            print_matrix_sequence("CPU 结果", C_cpu, N, M);
        }
        
        // 验证结果
        verify_results(C_gpu, C_cpu, M, N);
        
        // 释放内存
        free(A);
        free(B);
        free(C_gpu);
        free(C_cpu);
    }
    
    // 清理 cuBLAS
    destroy_cublas();
    printf("\n测试完成!\n");
    
    return 0;
}