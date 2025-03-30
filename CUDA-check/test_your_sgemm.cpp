#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <iostream>

using namespace std;


void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C);

//-------------------------------CUDA define-------------------------------------//

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define DEBUGs 1

bool use_gpu = true;

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
  //printf("初始化 cuBLAS...\n");
  if(cublas_handle == nullptr) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
      //fprintf(stderr, "无法获取 CUDA 设备数量: %s\n", cudaGetErrorString(err));
      return false;
    }
    
    if(device_count == 0) {
      //fprintf(stderr, "未找到 CUDA 设备\n");
      return false;
    }
    //("找到 %d 个 CUDA 设备\n", device_count);
    
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      //fprintf(stderr, "无法设置 CUDA 设备: %s\n", cudaGetErrorString(err));
      return false;
    }
    //printf("成功设置 CUDA 设备 0\n");
    
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      //fprintf(stderr, "无法创建 cuBLAS 句柄: %d\n", status);
      return false;
    }
    //printf("成功创建 cuBLAS 句柄\n");
  }

  return true;
}

void sgemm_cublas(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C)
{
    bool init_success = init_cublas();
    if (!init_success) {
        //fprintf(stderr, "cuBLAS 初始化失败\n");
        return;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, sizeof(float) * M * K);
    cudaMalloc((void **)&d_B, sizeof(float) * N * K);
    cudaMalloc((void **)&d_C, sizeof(float) * M * N);
    cudaMemcpy(d_A, A, M*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M*N * sizeof(float));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                M,
                N,
                K,
                &alpha,
                d_A,
                K,
                d_B,
                K,
                &beta,
                d_C,
                M);

    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //destroy_cublas();

}

// 优化的资源释放
void destroy_cublas() {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

//-------------------------------CUDA define-------------------------------------//


void getRandMatrix(float *matrix, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            matrix[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

void printMatrix(float *matrix, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void compareMatrices(float *matrix1, float *matrix2, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            if (fabs(matrix1[i * cols + j] - matrix2[i * cols + j]) > 1e-5) {
                printf("Matrices differ at (%ld, %ld): %f vs %f\n", i, j, matrix1[i * cols + j], matrix2[i * cols + j]);

            }
        }
    }
    //printf("Matrices are equal.\n");
}

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

void sgemm_check(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
    // 使用 CPU 计算结果
    float *C_cpu = (float *)malloc(M * N * sizeof(float));
    sgemm_cpu(M, N, K, A, B, C_cpu);

    // 使用 GPU 计算结果
    sgemm_cublas(M, N, K, A, B, C);

    // 打印 GPU 计算结果
    printf("GPU result:\n");
    printMatrix(C, M, N);

    // 打印 CPU 计算结果
    printf("CPU result:\n");
    printMatrix(C_cpu, M, N);

    // 比较 CPU 和 GPU 的结果
    compareMatrices(C_cpu, C, M, N);

    free(C_cpu);
}

int main()
{
    const int64_t M = 6;
    const int64_t N = 3;
    const int64_t K = 4;

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(N * K * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));

    getRandMatrix(A, M, K);
    getRandMatrix(B, N, K);

    //sgemm_check(M, N, K, A, B, C);

    float S[3][2] = { {1, 2}, {3, 4}, {5, 6} };
    float T[5][2] = { {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10} };
    float R[5][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},{0, 0, 0}};
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 3; j++) {
            for(int k = 0; k < 2; k++) {
                R[i][j] += S[j][k] * T[i][k];
            }
        }
    }

    printf("\nResult of matrix multiplication by cpu:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 3; j++) {
            printf("%f ", R[i][j]);
        }
        printf("\n");
    }

    sgemm_cublas(3, 5, 2, (float *)S, (float *)T, (float *)R);
    printf("\nResult of matrix multiplication by gpu:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 3; j++) {
            printf("%f ", R[i][j]);
        }
        printf("\n");
    }

    

    free(A);
    free(B);
    free(C);

    return 0;
}