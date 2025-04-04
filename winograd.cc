#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <iostream>

#include <thread>
#include <chrono>

#include <future>          
#include <functional> 

using namespace std;

#include "utils.h"

//-------------------------------cores define-------------------------------------//

const int64_t cores = omp_get_num_procs();
const int64_t threads_max = cores;
const int64_t threads_half = cores / 2;
const int64_t threads_quarter = cores / 4;
const int64_t threads_small = 4;
const int64_t threads_min = 2;

//-------------------------------cores define-------------------------------------//



void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C);

//-------------------------------CUDA define---------------------------------------//

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#define DEBUGs 1

bool use_gpu = true;
bool init_flag = false;

static int calledcount = 0;

// CUDA流
const int stream_count = 6;
static cudaStream_t g_stream = NULL;
static cudaStream_t g_streams[stream_count] = {nullptr};

static cudaEvent_t events[stream_count] = {nullptr};

// 定义 cublas 句柄
static cublasHandle_t cublas_handle = nullptr;
static cublasHandle_t cublas_handles[stream_count] = {nullptr};

// CUDA_ERROR CHECK
#define CHECK_CUDA_ERROR(call) { \
  cudaError_t err = call;      \
  if (err != cudaSuccess) {    \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      return false; \
  } \
}

// cuBLAS_ERROR CHECK
#define CHECK_CUBLAS_ERROR(call) { \
  cublasStatus_t status = call;  \
  if (status != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS error at %s:%d - code: %d\n", __FILE__, __LINE__, status); \
      return false; \
  } \
}

bool init_cublas() {
    
    //cudaSetDevice(0);
    //printf("成功设置 CUDA 设备 0\n");;

    if(stream_count > 0) {
        for (int i = 0; i < stream_count; i++) {
            cublasCreate(&cublas_handles[i]);
            cudaStreamCreate(&g_streams[i]);
            cudaEventCreate(&events[i]);
        }
    } 

    return true;
}

void sgemm_cublas(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C)
{

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

void destroy_cublas() {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

void destroy_multi_streams() {
  for (int i = 0; i < stream_count; i++) {
      if (cublas_handles[i] != NULL) {
          cublasDestroy(cublas_handles[i]);
          cublas_handles[i] = NULL;
      }
      
      if (g_streams[i] != NULL) {
          cudaStreamDestroy(g_streams[i]);
          g_streams[i] = NULL;
      }
  }
}

// 转换函数
void convert_float_to_half(float* src_f32, half* dst_f16, size_t size) {
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
      dst_f16[i] = __float2half(src_f32[i]);
  }
}


//-------------------------------CUDA define-------------------------------------//

//-------------------------------MEMORY define-------------------------------------//


// 内存池全局变量
static bool pool_initialized = false;

static bool mem_pre_allocated = false;

const unsigned long long init_memsize = 4000000000; // 4GB

// GPU内存
static half *g_d_A = nullptr;
static half *g_d_B = nullptr;
static half *g_d_C = nullptr;
static size_t g_d_A_size = 0;
static size_t g_d_B_size = 0;
static size_t g_d_C_size = 0;

static half *g_pinned_U_half = nullptr;
static half *g_pinned_V_half = nullptr;
static half *g_pinned_M_half = nullptr;
static size_t g_pinned_U_half_size = 0;
static size_t g_pinned_V_half_size = 0;
static size_t g_pinned_M_half_size = 0;

// 添加BF16全局变量-------------------------------
static __nv_bfloat16 *g_d_A_bf16 = nullptr;
static __nv_bfloat16 *g_d_B_bf16 = nullptr;
static __nv_bfloat16 *g_d_C_bf16 = nullptr;
static size_t g_d_A_bf16_size = 0;
static size_t g_d_B_bf16_size = 0;
static size_t g_d_C_bf16_size = 0;

static __nv_bfloat16 *g_pinned_U_bf16 = nullptr;
static __nv_bfloat16 *g_pinned_V_bf16 = nullptr;
static __nv_bfloat16 *g_pinned_M_bf16 = nullptr;
static size_t g_pinned_U_bf16_size = 0;
static size_t g_pinned_V_bf16_size = 0;
static size_t g_pinned_M_bf16_size = 0;
//----------------------------------------------


// Fp32

static float *total_pinned_memory = nullptr;

static float *g_pinned_U = nullptr;
static float *g_pinned_V = nullptr;
static float *g_pinned_M = nullptr;
static size_t g_pinned_U_size = 0;
static size_t g_pinned_V_size = 0;
static size_t g_pinned_M_size = 0;

// 写结合内存传输缓冲区
static float *g_pinned_transfer_V = nullptr;
static float *g_pinned_transfer_U = nullptr;
static size_t g_pinned_transfer_V_size = 0;
static size_t g_pinned_transfer_U_size = 0;


bool ensure_memory_size(void **mem, size_t *current_size, size_t required_size, bool is_pinned, bool write_combined = false) {
  //printf("current size: %zu, required size: %zu\n", *current_size, required_size);
  if (*current_size >= required_size ) {
      return true; 
  }
  
  //printf("Memory insufficient,current size: %zu, required size: %zu\n", *current_size, required_size);
  // 释放原有内存
  if (*mem) {
      if (is_pinned) {
          cudaFreeHost(*mem);
      } else {
          cudaFree(*mem);
      }
      *mem = nullptr;
  } 
  
  // 分配新内存
  cudaError_t err;
  if (is_pinned) {
      if (write_combined) {
          // 分配写结合内存
          err = cudaHostAlloc(mem, required_size, cudaHostAllocWriteCombined);
      } else {
          // 普通锁页内存
          err = cudaMallocHost(mem, required_size);
          //printf("Allocated pinned memory: %zu bytes\n", required_size);
      }
  } else {
      err = cudaMalloc(mem, required_size);
      //printf("Allocated device memory: %zu bytes\n", required_size);
  }
  
  if (err != cudaSuccess) {
      *current_size = 0;
      return false;
  }
  
  *current_size = required_size;
  return true;
}

// 预热CUDA环境和cuBLAS
__attribute__((constructor))
void warmup_cuda() {
    float *d_dummy;
    cudaMalloc(&d_dummy, 1024);
  
    cublasHandle_t temp_handle;
    cublasCreate(&temp_handle);
    
    // 执行一个小型矩阵乘法预热cuBLAS内核
    const int small_size = 128;
    const int precount = 10;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, small_size * small_size * sizeof(float));
    cudaMalloc(&d_B, small_size * small_size * sizeof(float));
    cudaMalloc(&d_C, small_size * small_size * sizeof(float));
    
    float alpha = 1.0f, beta = 0.0f;
    for(int i = 0; i < precount; i++){
      cublasSgemm(temp_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                small_size, small_size, small_size,
                &alpha, d_A, small_size, 
                d_B, small_size, 
                &beta, d_C, small_size);
    }
    
    
    cublasDestroy(temp_handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_dummy);
    
    if (!pool_initialized) {
        init_cublas();
        pool_initialized = true;
    }
    
    cudaDeviceSynchronize();
}

void __attribute__((destructor)) cleanup_memory_pool() {
  if (g_pinned_U) cudaFreeHost(g_pinned_U);
  if (g_pinned_V) cudaFreeHost(g_pinned_V);
  if (g_pinned_M) cudaFreeHost(g_pinned_M);
  if (g_pinned_U_half) cudaFreeHost(g_pinned_U_half);
  if (g_pinned_V_half) cudaFreeHost(g_pinned_V_half);
  if (g_pinned_M_half) cudaFreeHost(g_pinned_M_half);
  if (g_pinned_U_bf16) cudaFreeHost(g_pinned_U_bf16);
  if (g_pinned_V_bf16) cudaFreeHost(g_pinned_V_bf16);
  if (g_pinned_M_bf16) cudaFreeHost(g_pinned_M_bf16);
  if (g_d_A) cudaFree(g_d_A);
  if (g_d_B) cudaFree(g_d_B);
  if (g_d_C) cudaFree(g_d_C);
  if (g_d_A_bf16) cudaFree(g_d_A_bf16);
  if (g_d_B_bf16) cudaFree(g_d_B_bf16);
  if (g_d_C_bf16) cudaFree(g_d_C_bf16);
  if (g_stream) cudaStreamDestroy(g_stream);
  if (g_pinned_transfer_V) cudaFreeHost(g_pinned_transfer_V);
  if (g_pinned_transfer_U) cudaFreeHost(g_pinned_transfer_U);
  
  // 重置所有指针和大小
  g_pinned_U = g_pinned_V = g_pinned_M = nullptr;
  g_pinned_U_half = g_pinned_V_half = g_pinned_M_half = nullptr;
  g_pinned_U_bf16 = g_pinned_V_bf16 = g_pinned_M_bf16 = nullptr;
  g_d_A = g_d_B = g_d_C = nullptr;
  g_d_A_bf16 = g_d_B_bf16 = g_d_C_bf16 = nullptr;
  g_stream = nullptr;
  g_pinned_transfer_V = g_pinned_transfer_U = nullptr;
  g_pinned_transfer_V_size = g_pinned_transfer_U_size = 0;
  
  g_pinned_U_size = g_pinned_V_size = g_pinned_M_size = 0;
  g_pinned_U_half_size = g_pinned_V_half_size = g_pinned_M_half_size = 0;
  g_pinned_U_bf16_size = g_pinned_V_bf16_size = g_pinned_M_bf16_size = 0;
  g_d_A_size = g_d_B_size = g_d_C_size = 0;
  g_d_A_bf16_size = g_d_B_bf16_size = g_d_C_bf16_size = 0;
  
  pool_initialized = false;
}
//-----------------------------------MEMORY define---------------------------------//



struct alignas(64) parameters {
  float z0, z1, z2, z3, z4, z5, z6, z7; //内存对齐
};


void image_transform_256(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  //float z0, z1, z2, z3, z4, z5, z6;

  struct parameters zgroup;

  #pragma omp parallel for collapse(2) schedule(guided) private(zgroup) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
      for (int64_t w = 0; w < ti.tile_in_w; ++w) {

        //Non-SIMD Part
          zgroup.z6 = packed_image_tensor[0][w][idx];

          zgroup.z0 = 4.0f * zgroup.z6;

          zgroup.z6 = packed_image_tensor[1][w][idx];

          zgroup.z1 = -4.0f * zgroup.z6;
          zgroup.z2 = 4.0f * zgroup.z6;
          zgroup.z3 = -2.0f * zgroup.z6;
          zgroup.z4 = 2.0f * zgroup.z6;
          zgroup.z5 = 4.0f * zgroup.z6;

          zgroup.z6 = packed_image_tensor[2][w][idx];

          zgroup.z0 += -5.0f * zgroup.z6;
          zgroup.z1 += -4.0f * zgroup.z6;
          zgroup.z2 += -4.0f * zgroup.z6;
          zgroup.z3 += -zgroup.z6;
          zgroup.z4 += -zgroup.z6;

          zgroup.z6 = packed_image_tensor[3][w][idx];

          zgroup.z1 += zgroup.z6;
          zgroup.z2 += -zgroup.z6;
          zgroup.z3 += 2.0f * zgroup.z6;
          zgroup.z4 += -2.0f * zgroup.z6;
          zgroup.z5 += -5.0f * zgroup.z6;

          zgroup.z6 = packed_image_tensor[4][w][idx];

          zgroup.z0 += zgroup.z6;
          zgroup.z1 += zgroup.z6;
          zgroup.z2 += zgroup.z6;
          zgroup.z3 += zgroup.z6;
          zgroup.z4 += zgroup.z6;

          zgroup.z6 = packed_image_tensor[5][w][idx];

          zgroup.z5 += zgroup.z6;

          V_tensor[0][w][idx] = zgroup.z0;
          V_tensor[1][w][idx] = zgroup.z1;
          V_tensor[2][w][idx] = zgroup.z2;
          V_tensor[3][w][idx] = zgroup.z3;
          V_tensor[4][w][idx] = zgroup.z4;
          V_tensor[5][w][idx] = zgroup.z5;

      }

  }
        
  #pragma omp parallel for collapse(2) schedule(guided) private(zgroup) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++)
  {
      for (int64_t h = 0; h < ti.tile_in_h; ++h) {

        zgroup.z6 = V_tensor[h][0][idx];

        zgroup.z0 = 4.0f * zgroup.z6;

        zgroup.z6 = V_tensor[h][1][idx];

        zgroup.z1 = -4.0f * zgroup.z6;
        zgroup.z2 = 4.0f * zgroup.z6;
        zgroup.z3 = -2.0f * zgroup.z6;
        zgroup.z4 = 2.0f * zgroup.z6;
        zgroup.z5 = 4.0f * zgroup.z6;

        zgroup.z6 = V_tensor[h][2][idx];

        zgroup.z0 += -5.0f * zgroup.z6;
        zgroup.z1 += -4.0f * zgroup.z6;
        zgroup.z2 += -4.0f * zgroup.z6;
        zgroup.z3 += -zgroup.z6;
        zgroup.z4 += -zgroup.z6;

        zgroup.z6 = V_tensor[h][3][idx];

        zgroup.z1 += zgroup.z6;
        zgroup.z2 += -zgroup.z6;
        zgroup.z3 += 2.0f * zgroup.z6;
        zgroup.z4 += -2.0f * zgroup.z6;
        zgroup.z5 += -5.0f * zgroup.z6;

        zgroup.z6 = V_tensor[h][4][idx];

        zgroup.z0 += zgroup.z6;
        zgroup.z1 += zgroup.z6;
        zgroup.z2 += zgroup.z6;
        zgroup.z3 += zgroup.z6;
        zgroup.z4 += zgroup.z6;

        zgroup.z6 = V_tensor[h][5][idx];

        zgroup.z5 += zgroup.z6;

          V_tensor[h][0][idx] = zgroup.z0;
          V_tensor[h][1][idx] = zgroup.z1;
          V_tensor[h][2][idx] = zgroup.z2;
          V_tensor[h][3][idx] = zgroup.z3;
          V_tensor[h][4][idx] = zgroup.z4;
          V_tensor[h][5][idx] = zgroup.z5;
      }
  }
        
        
}

void image_transform(float *__restrict__ packed_image,
  float *__restrict__ V,
  const V_shape_t vs,
  const tiling_info_t ti,
  const int64_t collapsed_dim_size) {
typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
V_tensor_t V_tensor = (V_tensor_t)V;

    #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
        for (int64_t w = 0; w < ti.tile_in_w; ++w) {
            const int64_t block_size = (16LL < collapsed_dim_size - idx) ? 16 : (collapsed_dim_size - idx);

            __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

            __m512 vfour = _mm512_set1_ps(4.0f);
            __m512 vneg_four = _mm512_set1_ps(-4.0f);
            __m512 vneg_five = _mm512_set1_ps(-5.0f);
            __m512 vneg_two = _mm512_set1_ps(-2.0f);
            __m512 vtwo = _mm512_set1_ps(2.0f);
            __m512 vone = _mm512_set1_ps(1.0f);
            __m512 vneg_one = _mm512_set1_ps(-1.0f);


            __m512 vt0, vt1, vt2, vt3, vt4, vt5;

            __m512 vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[0][w][idx]);

            vt0 = _mm512_mul_ps(vfour, vz6);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[1][w][idx]);

            vt1 = _mm512_mul_ps(vneg_four, vz6);
            vt2 = _mm512_mul_ps(vfour, vz6);
            vt3 = _mm512_mul_ps(vneg_two, vz6);
            vt4 = _mm512_mul_ps(vtwo, vz6);
            vt5 = _mm512_mul_ps(vfour, vz6);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[2][w][idx]);

            vt0 = _mm512_fmadd_ps(vneg_five, vz6, vt0);
            vt1 = _mm512_fmadd_ps(vneg_four, vz6, vt1);
            vt2 = _mm512_fmadd_ps(vneg_four, vz6, vt2);
            vt3 = _mm512_fmadd_ps(vneg_one, vz6, vt3);
            vt4 = _mm512_fmadd_ps(vneg_one, vz6, vt4);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[3][w][idx]);

            vt1 = _mm512_fmadd_ps(vone, vz6, vt1);
            vt2 = _mm512_fmadd_ps(vneg_one, vz6, vt2);
            vt3 = _mm512_fmadd_ps(vtwo, vz6, vt3);
            vt4 = _mm512_fmadd_ps(vneg_two, vz6, vt4);
            vt5 = _mm512_fmadd_ps(vneg_five, vz6, vt5);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[4][w][idx]);

            vt0 = _mm512_add_ps(vt0, vz6);
            vt1 = _mm512_add_ps(vt1, vz6);
            vt2 = _mm512_add_ps(vt2, vz6);
            vt3 = _mm512_add_ps(vt3, vz6);
            vt4 = _mm512_add_ps(vt4, vz6);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[5][w][idx]);

            vt5 = _mm512_add_ps(vt5, vz6);

            _mm512_mask_storeu_ps(&V_tensor[0][w][idx], k_mask, vt0);
            _mm512_mask_storeu_ps(&V_tensor[1][w][idx], k_mask, vt1);
            _mm512_mask_storeu_ps(&V_tensor[2][w][idx], k_mask, vt2);
            _mm512_mask_storeu_ps(&V_tensor[3][w][idx], k_mask, vt3);
            _mm512_mask_storeu_ps(&V_tensor[4][w][idx], k_mask, vt4);
            _mm512_mask_storeu_ps(&V_tensor[5][w][idx], k_mask, vt5);
        }
    }

    #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
        for (int64_t h = 0; h < ti.tile_in_h; ++h) {
            const int64_t block_size = 16LL < collapsed_dim_size - idx ? 16 : (collapsed_dim_size - idx);
            __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

            __m512 vfour = _mm512_set1_ps(4.0f);
            __m512 vneg_four = _mm512_set1_ps(-4.0f);
            __m512 vneg_five = _mm512_set1_ps(-5.0f);
            __m512 vneg_two = _mm512_set1_ps(-2.0f);
            __m512 vtwo = _mm512_set1_ps(2.0f);
            __m512 vone = _mm512_set1_ps(1.0f);
            __m512 vneg_one = _mm512_set1_ps(-1.0f);

            __m512 vt0, vt1, vt2, vt3, vt4, vt5;

            __m512 vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][0][idx]);

            vt0 = _mm512_mul_ps(vfour, vz6);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][1][idx]);

            vt1 = _mm512_mul_ps(vneg_four, vz6);
            vt2 = _mm512_mul_ps(vfour, vz6);
            vt3 = _mm512_mul_ps(vneg_two, vz6);
            vt4 = _mm512_mul_ps(vtwo, vz6);
            vt5 = _mm512_mul_ps(vfour, vz6);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][2][idx]);

            vt0 = _mm512_fmadd_ps(vneg_five, vz6, vt0);
            vt1 = _mm512_fmadd_ps(vneg_four, vz6, vt1);
            vt2 = _mm512_fmadd_ps(vneg_four, vz6, vt2);
            vt3 = _mm512_fmadd_ps(vneg_one, vz6, vt3);
            vt4 = _mm512_fmadd_ps(vneg_one, vz6, vt4);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][3][idx]);

            vt1 = _mm512_fmadd_ps(vone, vz6, vt1);
            vt2 = _mm512_fmadd_ps(vneg_one, vz6, vt2);
            vt3 = _mm512_fmadd_ps(vtwo, vz6, vt3);
            vt4 = _mm512_fmadd_ps(vneg_two, vz6, vt4);
            vt5 = _mm512_fmadd_ps(vneg_five, vz6, vt5);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][4][idx]);

            vt0 = _mm512_add_ps(vt0, vz6);
            vt1 = _mm512_add_ps(vt1, vz6);
            vt2 = _mm512_add_ps(vt2, vz6);
            vt3 = _mm512_add_ps(vt3, vz6);
            vt4 = _mm512_add_ps(vt4, vz6);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][5][idx]);

            vt5 = _mm512_add_ps(vt5, vz6);

            _mm512_mask_storeu_ps(&V_tensor[h][0][idx], k_mask, vt0);
            _mm512_mask_storeu_ps(&V_tensor[h][1][idx], k_mask, vt1);
            _mm512_mask_storeu_ps(&V_tensor[h][2][idx], k_mask, vt2);
            _mm512_mask_storeu_ps(&V_tensor[h][3][idx], k_mask, vt3);
            _mm512_mask_storeu_ps(&V_tensor[h][4][idx], k_mask, vt4);
            _mm512_mask_storeu_ps(&V_tensor[h][5][idx], k_mask, vt5);
        }
    }
}


void filter_transform_256(float *__restrict__ packed_filter,
  float *__restrict__ U,
  const filter_shape_t fs,
  const U_shape_t us,
  const int64_t collapsed_dim_size) {
  typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  //float z0, z1, z2, z3, z4, z5, z6;
  struct parameters zgroup;

  //全部用zgroup
  #pragma omp parallel for collapse(2) schedule(guided) private(zgroup) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
        for (int64_t w = 0; w < fs.w; ++w){

          zgroup.z6 = packed_filter_tensor[w][0][idx];

          zgroup.z0 = (1.0f / 4.0f) * zgroup.z6;
          zgroup.z1 = (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 = (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 = (1.0f / 24.0f) * zgroup.z6;
          zgroup.z4 = (1.0f / 24.0f) * zgroup.z6;

          zgroup.z6 = packed_filter_tensor[w][1][idx];

          zgroup.z1 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 += (1.0f / 12.0f) * zgroup.z6;
          zgroup.z4 += (-1.0f / 12.0f) * zgroup.z6;

          zgroup.z6 = packed_filter_tensor[w][2][idx];

          zgroup.z1 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z4 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z5 = zgroup.z6;

          U_tensor[w][0][idx] = zgroup.z0;
          U_tensor[w][1][idx] = zgroup.z1;
          U_tensor[w][2][idx] = zgroup.z2;
          U_tensor[w][3][idx] = zgroup.z3;
          U_tensor[w][4][idx] = zgroup.z4;
          U_tensor[w][5][idx] = zgroup.z5;

        }

        
      }
      
  #pragma omp parallel for collapse(2) schedule(guided) private(zgroup) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++)
  {
      for (int64_t h = 0; h < us.h; ++h) {
          
          zgroup.z6 = U_tensor[0][h][idx];

          zgroup.z0 = (1.0f / 4.0f) * zgroup.z6;
          zgroup.z1 = (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 = (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 = (1.0f / 24.0f) * zgroup.z6;
          zgroup.z4 = (1.0f / 24.0f) * zgroup.z6;

          zgroup.z6 = U_tensor[1][h][idx];

          zgroup.z1 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 += (1.0f / 12.0f) * zgroup.z6;
          zgroup.z4 += (-1.0f / 12.0f) * zgroup.z6;

          zgroup.z6 = U_tensor[2][h][idx];

          zgroup.z1 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z4 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z5 = zgroup.z6;

          U_tensor[0][h][idx] = zgroup.z0;
          U_tensor[1][h][idx] = zgroup.z1;
          U_tensor[2][h][idx] = zgroup.z2;
          U_tensor[3][h][idx] = zgroup.z3;
          U_tensor[4][h][idx] = zgroup.z4;
          U_tensor[5][h][idx] = zgroup.z5;
      }
  }

        
}

void filter_transform(float *__restrict__ packed_filter,
  float *__restrict__ U,
  const filter_shape_t fs,
  const U_shape_t us,
  const int64_t collapsed_dim_size) {
typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
U_tensor_t U_tensor = (U_tensor_t)U;

#pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
      for (int64_t w = 0; w < fs.w; ++w) {
          const int64_t block_size = 16LL < collapsed_dim_size - idx ? 16 : (collapsed_dim_size - idx);
          __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

          __m512 vquarter = _mm512_set1_ps(1.0f / 4.0f);
          __m512 vneg_sixth = _mm512_set1_ps(-1.0f / 6.0f);
          __m512 vsixth = _mm512_set1_ps(1.0f / 6.0f);
          __m512 vtwentyFourth = _mm512_set1_ps(1.0f / 24.0f);
          __m512 vtwelfth = _mm512_set1_ps(1.0f / 12.0f);
          __m512 vneg_twelfth = _mm512_set1_ps(-1.0f / 12.0f);

          __m512 vt0, vt1, vt2, vt3, vt4, vt5;

          __m512 vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][0][idx]);

          vt0 = _mm512_mul_ps(vquarter, vz6);
          vt1 = _mm512_mul_ps(vneg_sixth, vz6);
          vt2 = _mm512_mul_ps(vneg_sixth, vz6);
          vt3 = _mm512_mul_ps(vtwentyFourth, vz6);
          vt4 = _mm512_mul_ps(vtwentyFourth, vz6);

          vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][1][idx]);

          vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
          vt2 = _mm512_fmadd_ps(vsixth, vz6, vt2);
          vt3 = _mm512_fmadd_ps(vtwelfth, vz6, vt3);
          vt4 = _mm512_fmadd_ps(vneg_twelfth, vz6, vt4);

          vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][2][idx]);

          vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
          vt2 = _mm512_fmadd_ps(vneg_sixth, vz6, vt2);
          vt3 = _mm512_fmadd_ps(vsixth, vz6, vt3);
          vt4 = _mm512_fmadd_ps(vsixth, vz6, vt4);
          vt5 = vz6;  

          _mm512_mask_storeu_ps(&U_tensor[w][0][idx], k_mask, vt0);
          _mm512_mask_storeu_ps(&U_tensor[w][1][idx], k_mask, vt1);
          _mm512_mask_storeu_ps(&U_tensor[w][2][idx], k_mask, vt2);
          _mm512_mask_storeu_ps(&U_tensor[w][3][idx], k_mask, vt3);
          _mm512_mask_storeu_ps(&U_tensor[w][4][idx], k_mask, vt4);
          _mm512_mask_storeu_ps(&U_tensor[w][5][idx], k_mask, vt5);
      }
}



#pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
      for (int64_t h = 0; h < us.h; ++h) {
            const int64_t block_size = 16LL < collapsed_dim_size - idx ? 16 : (collapsed_dim_size - idx);
            __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

            __m512 vquarter = _mm512_set1_ps(1.0f / 4.0f);
            __m512 vneg_sixth = _mm512_set1_ps(-1.0f / 6.0f);
            __m512 vsixth = _mm512_set1_ps(1.0f / 6.0f);
            __m512 vtwentyFourth = _mm512_set1_ps(1.0f / 24.0f);
            __m512 vtwelfth = _mm512_set1_ps(1.0f / 12.0f);
            __m512 vneg_twelfth = _mm512_set1_ps(-1.0f / 12.0f);

            __m512 vt0, vt1, vt2, vt3, vt4, vt5;

            __m512 vz6 = _mm512_maskz_loadu_ps(k_mask, &U_tensor[0][h][idx]);

            vt0 = _mm512_mul_ps(vquarter, vz6);
            vt1 = _mm512_mul_ps(vneg_sixth, vz6);
            vt2 = _mm512_mul_ps(vneg_sixth, vz6);
            vt3 = _mm512_mul_ps(vtwentyFourth, vz6);
            vt4 = _mm512_mul_ps(vtwentyFourth, vz6);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &U_tensor[1][h][idx]);

            vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
            vt2 = _mm512_fmadd_ps(vsixth, vz6, vt2);
            vt3 = _mm512_fmadd_ps(vtwelfth, vz6, vt3);
            vt4 = _mm512_fmadd_ps(vneg_twelfth, vz6, vt4);

            vz6 = _mm512_maskz_loadu_ps(k_mask, &U_tensor[2][h][idx]);

            vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
            vt2 = _mm512_fmadd_ps(vneg_sixth, vz6, vt2);
            vt3 = _mm512_fmadd_ps(vsixth, vz6, vt3);
            vt4 = _mm512_fmadd_ps(vsixth, vz6, vt4);
            vt5 = vz6;

            _mm512_mask_storeu_ps(&U_tensor[0][h][idx], k_mask, vt0);
            _mm512_mask_storeu_ps(&U_tensor[1][h][idx], k_mask, vt1);
            _mm512_mask_storeu_ps(&U_tensor[2][h][idx], k_mask, vt2);
            _mm512_mask_storeu_ps(&U_tensor[3][h][idx], k_mask, vt3);
            _mm512_mask_storeu_ps(&U_tensor[4][h][idx], k_mask, vt4);
            _mm512_mask_storeu_ps(&U_tensor[5][h][idx], k_mask, vt5);
      }
  }
}

void output_transform(float *__restrict__ M,
  float *__restrict__ Y,
  const tiling_info_t ti,
  const int64_t collapsed_dim_size) {
typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
M_tensor_t M_tensor = (M_tensor_t)M;
Y_tensor_t Y_tensor = (Y_tensor_t)Y;


  #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
      for (int64_t w = 0; w < ti.tile_in_w; ++w) {
          const int64_t block_size = std::min<int64_t>(16, collapsed_dim_size - idx);
          __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

          __m512 vneg_one = _mm512_set1_ps(-1.0f);
          __m512 vtwo = _mm512_set1_ps(2.0f);
          __m512 vfour = _mm512_set1_ps(4.0f);
          __m512 veight = _mm512_set1_ps(8.0f);
          __m512 vneg_two = _mm512_set1_ps(-2.0f);
          __m512 vneg_eight = _mm512_set1_ps(-8.0f);

          __m512 vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[0][w][idx]);

          __m512 vr0 = vz4;

          vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[1][w][idx]);

          vr0 = _mm512_add_ps(vr0, vz4);
          __m512 vr1 = vz4;
          __m512 vr2 = vz4;
          __m512 vr3 = vz4;

          vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[2][w][idx]);

          vr0 = _mm512_add_ps(vr0, vz4);
          vr1 = _mm512_fmadd_ps(vneg_one, vz4, vr1);
          vr2 = _mm512_add_ps(vr2, vz4);
          vr3 = _mm512_fmadd_ps(vneg_one, vz4, vr3);

          vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[3][w][idx]);

          vr0 = _mm512_add_ps(vr0, vz4);
          vr1 = _mm512_fmadd_ps(vtwo, vz4, vr1);
          vr2 = _mm512_fmadd_ps(vfour, vz4, vr2);
          vr3 = _mm512_fmadd_ps(veight, vz4, vr3);

          vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[4][w][idx]);

          vr0 = _mm512_add_ps(vr0, vz4);
          vr1 = _mm512_fmadd_ps(vneg_two, vz4, vr1);
          vr2 = _mm512_fmadd_ps(vfour, vz4, vr2);
          vr3 = _mm512_fmadd_ps(vneg_eight, vz4, vr3);

          vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[5][w][idx]);

          vr3 = _mm512_add_ps(vr3, vz4);

          _mm512_mask_storeu_ps(&Y_tensor[0][w][idx], k_mask, vr0);
          _mm512_mask_storeu_ps(&Y_tensor[1][w][idx], k_mask, vr1);
          _mm512_mask_storeu_ps(&Y_tensor[2][w][idx], k_mask, vr2);
          _mm512_mask_storeu_ps(&Y_tensor[3][w][idx], k_mask, vr3);
      }
  }

  #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
        for (int64_t h = 0; h < ti.tile_out_h; ++h) {
            const int64_t block_size = std::min<int64_t>(16, collapsed_dim_size - idx);
            __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

            __m512 vneg_one = _mm512_set1_ps(-1.0f);
            __m512 vtwo = _mm512_set1_ps(2.0f);
            __m512 vfour = _mm512_set1_ps(4.0f);
            __m512 veight = _mm512_set1_ps(8.0f);
            __m512 vneg_two = _mm512_set1_ps(-2.0f);
            __m512 vneg_eight = _mm512_set1_ps(-8.0f);

            __m512 vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][0][idx]);

            __m512 vr0 = vz4;

            vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][1][idx]);

            vr0 = _mm512_add_ps(vr0, vz4);
            __m512 vr1 = vz4;
            __m512 vr2 = vz4;
            __m512 vr3 = vz4;

            vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][2][idx]);

            vr0 = _mm512_add_ps(vr0, vz4);
            vr1 = _mm512_fmadd_ps(vneg_one, vz4, vr1);
            vr2 = _mm512_add_ps(vr2, vz4);
            vr3 = _mm512_fmadd_ps(vneg_one, vz4, vr3);

            vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][3][idx]);

            vr0 = _mm512_add_ps(vr0, vz4);
            vr1 = _mm512_fmadd_ps(vtwo, vz4, vr1);
            vr2 = _mm512_fmadd_ps(vfour, vz4, vr2);
            vr3 = _mm512_fmadd_ps(veight, vz4, vr3);

            vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][4][idx]);

            vr0 = _mm512_add_ps(vr0, vz4);
            vr1 = _mm512_fmadd_ps(vneg_two, vz4, vr1);
            vr2 = _mm512_fmadd_ps(vfour, vz4, vr2);
            vr3 = _mm512_fmadd_ps(vneg_eight, vz4, vr3);

            vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][5][idx]);

            vr3 = _mm512_add_ps(vr3, vz4);

            _mm512_mask_storeu_ps(&Y_tensor[h][0][idx], k_mask, vr0);
            _mm512_mask_storeu_ps(&Y_tensor[h][1][idx], k_mask, vr1);
            _mm512_mask_storeu_ps(&Y_tensor[h][2][idx], k_mask, vr2);
            _mm512_mask_storeu_ps(&Y_tensor[h][3][idx], k_mask, vr3);
        }
    }
}


void output_transform_256(float *__restrict__ M,
                      float *__restrict__ Y,
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size) {
  typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  //float z0, z1, z2, z3, z4;
  struct parameters zgroup;

  #pragma omp parallel for collapse(2) schedule(guided) private(zgroup) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      zgroup.z4 = M_tensor[0][w][idx];

      zgroup.z0 = zgroup.z4;

      zgroup.z4 = M_tensor[1][w][idx];
      zgroup.z0 += zgroup.z4;
      zgroup.z1 = zgroup.z4;
      zgroup.z2 = zgroup.z4;
      zgroup.z3 = zgroup.z4;

      zgroup.z4 = M_tensor[2][w][idx];
      zgroup.z0 += zgroup.z4;
      zgroup.z1 += -zgroup.z4;
      zgroup.z2 += zgroup.z4;
      zgroup.z3 += -zgroup.z4;

      zgroup.z4 = M_tensor[3][w][idx];
      zgroup.z0 += zgroup.z4;
      zgroup.z1 += 2.0f * zgroup.z4;
      zgroup.z2 += 4.0f * zgroup.z4;
      zgroup.z3 += 8.0f * zgroup.z4;

      zgroup.z4 = M_tensor[4][w][idx];
      zgroup.z0 += zgroup.z4;
      zgroup.z1 += -2.0f * zgroup.z4;
      zgroup.z2 += 4.0f * zgroup.z4;
      zgroup.z3 += -8.0f * zgroup.z4;

      zgroup.z4 = M_tensor[5][w][idx];

      zgroup.z3 += zgroup.z4;

      Y_tensor[0][w][idx] = zgroup.z0;
      Y_tensor[1][w][idx] = zgroup.z1;
      Y_tensor[2][w][idx] = zgroup.z2;
      Y_tensor[3][w][idx] = zgroup.z3;

    } 
  }

  //h和idx交换顺序，当内层循环完成一个，就生成以个batch的图片，然后立刻导入gpu，实现流水线
  #pragma omp parallel for collapse(2) schedule(guided) private(zgroup) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++){
    for (int64_t h = 0; h < ti.tile_out_h; ++h) {

      zgroup.z4 = Y_tensor[h][0][idx];

      zgroup.z0 = zgroup.z4;

      zgroup.z4 = Y_tensor[h][1][idx];
      zgroup.z0 += zgroup.z4;
      zgroup.z1 = zgroup.z4;
      zgroup.z2 = zgroup.z4;
      zgroup.z3 = zgroup.z4;

      zgroup.z4 = Y_tensor[h][2][idx];
      zgroup.z0 += zgroup.z4;
      zgroup.z1 += -zgroup.z4;
      zgroup.z2 += zgroup.z4;
      zgroup.z3 += -zgroup.z4;

      zgroup.z4 = Y_tensor[h][3][idx];
      zgroup.z0 += zgroup.z4;
      zgroup.z1 += 2.0f * zgroup.z4;
      zgroup.z2 += 4.0f * zgroup.z4;
      zgroup.z3 += 8.0f * zgroup.z4;

      zgroup.z4 = Y_tensor[h][4][idx];
      zgroup.z0 += zgroup.z4;
      zgroup.z1 += -2.0f * zgroup.z4;
      zgroup.z2 += 4.0f * zgroup.z4;
      zgroup.z3 += -8.0f * zgroup.z4;

      zgroup.z4 = Y_tensor[h][5][idx];

      zgroup.z3 += zgroup.z4;

      Y_tensor[h][0][idx] = zgroup.z0;
      Y_tensor[h][1][idx] = zgroup.z1;
      Y_tensor[h][2][idx] = zgroup.z2;
      Y_tensor[h][3][idx] = zgroup.z3;

    }
  }
    
}

////////////////////////////////////////pipeline//////////////////////////////////////////////

//第一次变换，无法分批处理
void image_transform_stage1(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  //float z0, z1, z2, z3, z4, z5, z6;

  struct parameters zgroup;

  #pragma omp parallel for collapse(2) schedule(guided) private(zgroup) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
      for (int64_t w = 0; w < ti.tile_in_w; ++w) {

        //Non-SIMD Part
          zgroup.z6 = packed_image_tensor[0][w][idx];

          zgroup.z0 = 4.0f * zgroup.z6;

          zgroup.z6 = packed_image_tensor[1][w][idx];

          zgroup.z1 = -4.0f * zgroup.z6;
          zgroup.z2 = 4.0f * zgroup.z6;
          zgroup.z3 = -2.0f * zgroup.z6;
          zgroup.z4 = 2.0f * zgroup.z6;
          zgroup.z5 = 4.0f * zgroup.z6;

          zgroup.z6 = packed_image_tensor[2][w][idx];

          zgroup.z0 += -5.0f * zgroup.z6;
          zgroup.z1 += -4.0f * zgroup.z6;
          zgroup.z2 += -4.0f * zgroup.z6;
          zgroup.z3 += -zgroup.z6;
          zgroup.z4 += -zgroup.z6;

          zgroup.z6 = packed_image_tensor[3][w][idx];

          zgroup.z1 += zgroup.z6;
          zgroup.z2 += -zgroup.z6;
          zgroup.z3 += 2.0f * zgroup.z6;
          zgroup.z4 += -2.0f * zgroup.z6;
          zgroup.z5 += -5.0f * zgroup.z6;

          zgroup.z6 = packed_image_tensor[4][w][idx];

          zgroup.z0 += zgroup.z6;
          zgroup.z1 += zgroup.z6;
          zgroup.z2 += zgroup.z6;
          zgroup.z3 += zgroup.z6;
          zgroup.z4 += zgroup.z6;

          zgroup.z6 = packed_image_tensor[5][w][idx];

          zgroup.z5 += zgroup.z6;

          V_tensor[0][w][idx] = zgroup.z0;
          V_tensor[1][w][idx] = zgroup.z1;
          V_tensor[2][w][idx] = zgroup.z2;
          V_tensor[3][w][idx] = zgroup.z3;
          V_tensor[4][w][idx] = zgroup.z4;
          V_tensor[5][w][idx] = zgroup.z5;

      }

  }
}

//第二阶段，根据stream=6，调用一次处理一行
void image_transform_stage2(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size,
                     const int64_t offset) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  struct parameters zgroup;

  //计算特定流的偏移量
  int64_t h_offset = offset;  
  
  //处理完一整行的数据后，立刻将结果存入GPU
  for(int idx = 0; idx < collapsed_dim_size; idx++){

    zgroup.z6 = V_tensor[h_offset][0][idx];

    zgroup.z0 = 4.0f * zgroup.z6;

    zgroup.z6 = V_tensor[h_offset][1][idx];

    zgroup.z1 = -4.0f * zgroup.z6;
    zgroup.z2 = 4.0f * zgroup.z6;
    zgroup.z3 = -2.0f * zgroup.z6;
    zgroup.z4 = 2.0f * zgroup.z6;
    zgroup.z5 = 4.0f * zgroup.z6;

    zgroup.z6 = V_tensor[h_offset][2][idx];

    zgroup.z0 += -5.0f * zgroup.z6;
    zgroup.z1 += -4.0f * zgroup.z6;
    zgroup.z2 += -4.0f * zgroup.z6;
    zgroup.z3 += -zgroup.z6;
    zgroup.z4 += -zgroup.z6;

    zgroup.z6 = V_tensor[h_offset][3][idx];

    zgroup.z1 += zgroup.z6;
    zgroup.z2 += -zgroup.z6;
    zgroup.z3 += 2.0f * zgroup.z6;
    zgroup.z4 += -2.0f * zgroup.z6;
    zgroup.z5 += -5.0f * zgroup.z6;

    zgroup.z6 = V_tensor[h_offset][4][idx];

    zgroup.z0 += zgroup.z6;
    zgroup.z1 += zgroup.z6;
    zgroup.z2 += zgroup.z6;
    zgroup.z3 += zgroup.z6;
    zgroup.z4 += zgroup.z6;

    zgroup.z6 = V_tensor[h_offset][5][idx];

    zgroup.z5 += zgroup.z6;

    V_tensor[h_offset][0][idx] = zgroup.z0;
    V_tensor[h_offset][1][idx] = zgroup.z1;
    V_tensor[h_offset][2][idx] = zgroup.z2;
    V_tensor[h_offset][3][idx] = zgroup.z3;
    V_tensor[h_offset][4][idx] = zgroup.z4;
    V_tensor[h_offset][5][idx] = zgroup.z5;

  }
  //处理完一整行的数据后，立刻将结果存入GPU
}


//filter
void filter_transform_stage1(float *__restrict__ packed_filter,
  float *__restrict__ U,
  const filter_shape_t fs,
  const U_shape_t us,
  const int64_t collapsed_dim_size) {
  typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  //float z0, z1, z2, z3, z4, z5, z6;
  struct parameters zgroup;

  //全部用zgroup
  #pragma omp parallel for collapse(2) schedule(guided) private(zgroup) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
        for (int64_t w = 0; w < fs.w; ++w){

          zgroup.z6 = packed_filter_tensor[w][0][idx];

          zgroup.z0 = (1.0f / 4.0f) * zgroup.z6;
          zgroup.z1 = (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 = (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 = (1.0f / 24.0f) * zgroup.z6;
          zgroup.z4 = (1.0f / 24.0f) * zgroup.z6;

          zgroup.z6 = packed_filter_tensor[w][1][idx];

          zgroup.z1 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 += (1.0f / 12.0f) * zgroup.z6;
          zgroup.z4 += (-1.0f / 12.0f) * zgroup.z6;

          zgroup.z6 = packed_filter_tensor[w][2][idx];

          zgroup.z1 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z2 += (-1.0f / 6.0f) * zgroup.z6;
          zgroup.z3 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z4 += (1.0f / 6.0f) * zgroup.z6;
          zgroup.z5 = zgroup.z6;

          U_tensor[w][0][idx] = zgroup.z0;
          U_tensor[w][1][idx] = zgroup.z1;
          U_tensor[w][2][idx] = zgroup.z2;
          U_tensor[w][3][idx] = zgroup.z3;
          U_tensor[w][4][idx] = zgroup.z4;
          U_tensor[w][5][idx] = zgroup.z5;

        } 
      }
}

void filter_transform_stage2(float *__restrict__ packed_filter,
  float *__restrict__ U,
  const filter_shape_t fs,
  const U_shape_t us,
  const int64_t collapsed_dim_size,
  const int64_t offset) {
  typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  struct parameters zgroup;

  //计算特定流的偏移量
  int64_t h_offset = offset;  

  //printf("Filiter transform stage2 called for stream %d   ", offset);

  //处理完一整行的数据后，立刻将结果存入GPU
  for(int idx = 0; idx < collapsed_dim_size; idx++){

    zgroup.z6 = U_tensor[0][h_offset][idx];

    zgroup.z0 = (1.0f / 4.0f) * zgroup.z6;
    zgroup.z1 = (-1.0f / 6.0f) * zgroup.z6;
    zgroup.z2 = (-1.0f / 6.0f) * zgroup.z6;
    zgroup.z3 = (1.0f / 24.0f) * zgroup.z6;
    zgroup.z4 = (1.0f / 24.0f) * zgroup.z6;

    zgroup.z6 = U_tensor[1][h_offset][idx];

    zgroup.z1 += (-1.0f / 6.0f) * zgroup.z6;
    zgroup.z2 += (1.0f / 6.0f) * zgroup.z6;
    zgroup.z3 += (1.0f / 12.0f) * zgroup.z6;
    zgroup.z4 += (-1.0f / 12.0f) * zgroup.z6;

    zgroup.z6 = U_tensor[2][h_offset][idx];

    zgroup.z1 += (-1.0f / 6.0f) * zgroup.z6;
    zgroup.z2 += (-1.0f / 6.0f) * zgroup.z6;
    zgroup.z3 += (1.0f / 6.0f) * zgroup.z6;
    zgroup.z4 += (1.0f / 6.0f) * zgroup.z6;
    zgroup.z5 = zgroup.z6;

    U_tensor[0][h_offset][idx] = zgroup.z0;
    U_tensor[1][h_offset][idx] = zgroup.z1;
    U_tensor[2][h_offset][idx] = zgroup.z2;
    U_tensor[3][h_offset][idx] = zgroup.z3;
    U_tensor[4][h_offset][idx] = zgroup.z4;
    U_tensor[5][h_offset][idx] = zgroup.z5;
  }

  //printf("---Called ended\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////

void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

  #pragma omp parallel for collapse(4) schedule(guided) 
    for (int64_t h = 0; h < fs.h; ++h)
      for (int64_t w = 0; w < fs.w; ++w)
        for (int64_t oc = 0; oc < fs.oc; oc++)
          for (int64_t ic = 0; ic < fs.ic; ic++)
            packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];
}


void image_packing(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
  typedef float(*packedImage_tensor_t)[ti.tile_in_w][ti.num_tiles][is.ic];
  typedef float(*image_tensor_t)[is.ic][is.h][is.w];
  packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
  image_tensor_t image_tensor = (image_tensor_t)image;

  #pragma omp parallel for collapse(4) schedule(guided) 
  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
    for (int64_t ic = 0; ic < is.ic; ic++) {
      for (int64_t h = 0; h < ti.tile_in_h; ++h) {
        for (int64_t w = 0; w < ti.tile_in_w; ++w) {
              tile_index_t tidx = get_tile_index(tile, ti);
              int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
              if (hh * 4 + h < is.h && ww * 4 + w < is.w)
                packed_image_tensor[h][w][tile][ic] = image_tensor[batch][ic][(hh * 4 + h)][(ww * 4 + w)];
              else
                packed_image_tensor[h][w][tile][ic] = 0;
            }
          }
    }
  }
}

void output_unpacking_store(float *__restrict__ Y,
                            float *__restrict__ out,
                            const out_shape_t os,
                            const tiling_info_t ti) {
  typedef float(*Y_tensor_t)[ti.tile_in_w][os.oc][ti.num_tiles];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;

  #pragma omp parallel for collapse(4) schedule(guided) 
  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    for (int64_t w = 0; w < ti.tile_out_w; ++w) {
      for (int64_t oc = 0; oc < os.oc; oc++) {
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if (hh * 4 + h < os.h && ww * 4 + w < os.w)
            out_tensor[batch][oc][(hh * 4 + h)][(ww * 4 + w)] = Y_tensor[h][w][oc][tile];
        }
      }
    }
  }
}

void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
  typedef float(*A_tensor_t)[K];
  typedef float(*B_tensor_t)[K];
  typedef float(*C_tensor_t)[M];
  A_tensor_t A_tensor = (A_tensor_t)A;
  B_tensor_t B_tensor = (B_tensor_t)B;
  C_tensor_t C_tensor = (C_tensor_t)C;

  const int BM = 32; // M方向分块
  const int BN = 32; // N方向分块
  const int BK = 32; // K方向分块

  for (int64_t bn = 0; bn < N; bn += BN) {
    for (int64_t bm = 0; bm < M; bm += BM) {
      const int64_t max_n = std::min(bn + BN, N);
      const int64_t max_m = std::min(bm + BM, M);
      
      // 初始化当前分块的C
      for (int64_t n = bn; n < max_n; ++n) {
        for (int64_t m = bm; m < max_m; ++m) {
          C_tensor[n][m] = 0.0f;
        }
      }
      
      // K方向分块处理
      for (int64_t bk = 0; bk < K; bk += BK) {
        const int64_t max_k = std::min(bk + BK, K);
        
        for (int64_t n = bn; n < max_n; ++n) {
          for (int64_t m = bm; m < max_m; ++m) {
            __m256 sum = _mm256_setzero_ps();
            
            // 向量化内层循环
            for (int64_t k = bk; k < max_k - 7; k += 8) {
              __m256 a_vec = _mm256_loadu_ps(&A_tensor[m][k]);
              __m256 b_vec = _mm256_loadu_ps(&B_tensor[n][k]);
              sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
            }
            
            alignas(32) float result[8];
            _mm256_store_ps(result, sum);
            float final_sum = 0.0f;
            for (int i = 0; i < 8; i++) {
              final_sum += result[i];
            }
            
            // 处理剩余元素
            for (int64_t k = bk + ((max_k - bk) / 8) * 8; k < max_k; ++k) {
              final_sum += A_tensor[m][k] * B_tensor[n][k];
            }
            
            C_tensor[n][m] += final_sum;
          }
        }
      }
    }
  }
}

void winograd_convolution(float *__restrict__ image, const int image_height,
  const int image_width, const int input_channel_num,
  float *__restrict__ filter, const int output_channel_num,
  const int batch_num, float *__restrict__ out) {
    // 基本数据准备保持不变
    const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
    const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
    const out_shape_t os = get_output_shape(is, fs);
    const tiling_info_t ti = get_tiling_info(is, os);
    const U_shape_t us = get_U_shape(fs, ti);
    const V_shape_t vs = get_V_shape(is, ti);

    int batch_size = ti.tile_in_h * ti.tile_in_w;
    const int m = vs.num_tiles;  // 输出矩阵的行数
    const int n = us.oc;        // 输出矩阵的列数
    const int k = us.ic;        // 内部维度

    const int cpu_limit = 256 * 256 * 512; // CPU限制
    if(m*n*k < cpu_limit)
    {
        //printf("CPU\n");
        float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
        float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
        float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
        float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
        float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
        float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);
        
          filter_packing(filter, packed_filter, fs);
          filter_transform(packed_filter, U, fs, us, us.oc * us.ic);
        
          image_packing(image, packed_image, is, ti);
          image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);
      
      //CUDA Prepare
        #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
        for (int64_t h = 0; h < ti.tile_in_h; ++h) {
            for (int64_t w = 0; w < ti.tile_in_w; ++w) {
                typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][us.ic];
                typedef float(*V_tensor_t)[ti.tile_in_w][vs.num_tiles][vs.ic];
                typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
                U_tensor_t U_tensor = (U_tensor_t)U;
                V_tensor_t V_tensor = (V_tensor_t)V;
                M_tensor_t M_tensor = (M_tensor_t)M;
            
                // 初始化 M_tensor[h][w] 为 0
                //memset(M_tensor[h][w], 0, sizeof(float) * us.oc * vs.num_tiles);
                sgemm(vs.num_tiles,
                  us.oc,
                  us.ic,
                  (float *)(V_tensor[h][w]),
                  (float *)(U_tensor[h][w]),
                  (float *)(M_tensor[h][w]));
    
              }
          
        }
            
        output_transform(M, Y, ti, us.oc * vs.num_tiles);
        output_unpacking_store(Y, out, os, ti);
        
        //destroy_cublas();
      
        free(packed_filter);
        free(packed_image);
        free(U);
        free(V);
        free(M);
        free(Y);

        return;
    }


    
    else{

        //printf("CUDA\n");
          // 计算所需内存大小
        const long long A_size = vs.num_tiles * vs.ic;  // V矩阵大小
        const long long B_size = us.oc * us.ic;        // U矩阵大小
        const long long C_size = vs.num_tiles * us.oc;  // M矩阵大小
        
        size_t pinned_U_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic;
        size_t pinned_V_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic;
        size_t pinned_M_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles;
        
        size_t d_A_req_size = sizeof(float) * batch_size * A_size;
        size_t d_B_req_size = sizeof(float) * batch_size * B_size;
        size_t d_C_req_size = sizeof(float) * batch_size * C_size;
        
        // 初始化 cuBLAS (如果还没有初始化)
        if(!init_flag) {
            init_cublas();
            init_flag = true;
        }
        
        // 初始化内存池（如果是第一次使用）
        if (!pool_initialized) {
            // 创建CUDA流
            if (g_stream == NULL && cudaStreamCreate(&g_stream) != cudaSuccess) {
                fprintf(stderr, "Failed to create CUDA stream\n");
                return;
            }
            
             // 预分配所有内存，确保大小足够
             if(mem_pre_allocated)
             {
                ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, init_memsize, true);
                ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, init_memsize, true);
                ensure_memory_size((void**)&g_pinned_M, &g_pinned_M_size, init_memsize, true);
                ensure_memory_size((void**)&g_d_A, &g_d_A_size, init_memsize, false);
                ensure_memory_size((void**)&g_d_B, &g_d_B_size, init_memsize, false);
                ensure_memory_size((void**)&g_d_C, &g_d_C_size, init_memsize, false);
                //printf("Initialized memory pool with size: %lld\n", init_memsize);
             }
            
            // 标记为已初始化
            pool_initialized = true;
        }
        
        // 确保所有内存大小足够
        bool memory_ok = 
            ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, pinned_U_req_size, true) &&
            ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, pinned_V_req_size, true) &&
            ensure_memory_size((void**)&g_pinned_M, &g_pinned_M_size, pinned_M_req_size, true) &&
            ensure_memory_size((void**)&g_d_A, &g_d_A_size, d_A_req_size, false) &&
            ensure_memory_size((void**)&g_d_B, &g_d_B_size, d_B_req_size, false) &&
            ensure_memory_size((void**)&g_d_C, &g_d_C_size, d_C_req_size, false);
        
        
        // 设置 cublas 流
        cublasSetStream(cublas_handle, g_stream);

        // 普通内存分配（这些较小，可以每次重新分配）
        float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
        float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
        float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

        // 准备数据阶段保持不变
        filter_packing(filter, packed_filter, fs);
        filter_transform(packed_filter, g_pinned_U, fs, us, us.oc * us.ic);

        image_packing(image, packed_image, is, ti);
        image_transform(packed_image, g_pinned_V, vs, ti, vs.ic * vs.num_tiles);

        // 使用异步内存复制将数据传输到GPU
        cudaMemcpyAsync(g_d_A, g_pinned_V, batch_size * A_size * sizeof(float), 
                      cudaMemcpyHostToDevice, g_stream);
        cudaMemcpyAsync(g_d_B, g_pinned_U, batch_size * B_size * sizeof(float), 
                      cudaMemcpyHostToDevice, g_stream);

        // 步长 - 每个矩阵的大小（元素数量）
        long long strideA = A_size;
        long long strideB = B_size;
        long long strideC = C_size;

        // 执行批处理矩阵乘法
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // 使用带步长的批处理GEMM (在同一流中执行)
        cublasGemmStridedBatchedEx(
            cublas_handle,
            CUBLAS_OP_T,     // A转置
            CUBLAS_OP_N,     // B不转置
            m,               // 矩阵C的行数(vs.num_tiles)
            n,               // 矩阵C的列数(us.oc)
            k,               // 内部维度(us.ic)
            &alpha,          // 缩放因子
            g_d_A,           // V矩阵起始地址
            CUDA_R_32F,      // 数据类型:float
            k,               // V矩阵的前导维度
            strideA,         // V矩阵序列的步长
            g_d_B,           // U矩阵起始地址
            CUDA_R_32F,      // 数据类型:float
            k,               // U矩阵的前导维度
            strideB,         // U矩阵序列的步长
            &beta,           // 缩放因子
            g_d_C,           // M矩阵起始地址
            CUDA_R_32F,      // 数据类型:float
            m,               // M矩阵的前导维度
            strideC,         // M矩阵序列的步长
            batch_size,      // 批次数量
            CUDA_R_32F,      // 计算类型:float
            CUBLAS_GEMM_DEFAULT // 使用默认算法
        );

        // 异步将结果复制回主机（使用页锁定内存）
        cudaMemcpyAsync(g_pinned_M, g_d_C, batch_size * C_size * sizeof(float), 
                      cudaMemcpyDeviceToHost, g_stream);
        
        // 在流上同步，确保传输完成
        cudaStreamSynchronize(g_stream);
        
        // 输出处理保持不变
        output_transform(g_pinned_M, Y, ti, us.oc * vs.num_tiles);
        output_unpacking_store(Y, out, os, ti);

        // 释放普通内存
        free(packed_filter);
        free(packed_image);
        free(Y);
        
    }

}


//------------------------------------------------Finished--------------------------------------------------//

void winograd_convolution_multistream(float *__restrict__ image, const int image_height,
  const int image_width, const int input_channel_num,
  float *__restrict__ filter, const int output_channel_num,
  const int batch_num, float *__restrict__ out) {

    const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
    const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
    const out_shape_t os = get_output_shape(is, fs);
    const tiling_info_t ti = get_tiling_info(is, os);
    const U_shape_t us = get_U_shape(fs, ti);
    const V_shape_t vs = get_V_shape(is, ti);

    const int m = vs.num_tiles;  // 输出矩阵的行数
    const int n = us.oc;        // 输出矩阵的列数
    const int k = us.ic;    // 内部维度

    const int cpu_limit = 256 *256 * 512; // CPU限制
    if(m*n*k < cpu_limit)
    {
        //printf("CPU\n");
        float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
        float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
        float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
        float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
        float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
        float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);
          
        filter_packing(filter, packed_filter, fs);
        filter_transform(packed_filter, U, fs, us, us.oc * us.ic);
        
        image_packing(image, packed_image, is, ti);
        image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

      //CUDA Prepare
        #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
        for (int64_t h = 0; h < ti.tile_in_h; ++h) {
            for (int64_t w = 0; w < ti.tile_in_w; ++w) {
                typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][us.ic];
                typedef float(*V_tensor_t)[ti.tile_in_w][vs.num_tiles][vs.ic];
                typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
                U_tensor_t U_tensor = (U_tensor_t)U;
                V_tensor_t V_tensor = (V_tensor_t)V;
                M_tensor_t M_tensor = (M_tensor_t)M;

                sgemm(vs.num_tiles,
                  us.oc,
                  us.ic,
                  (float *)(V_tensor[h][w]),
                  (float *)(U_tensor[h][w]),
                  (float *)(M_tensor[h][w]));
    
              }
          
        }
            
        output_transform(M, Y, ti, us.oc * vs.num_tiles);
        output_unpacking_store(Y, out, os, ti);
        
        //destroy_cublas();
      
        free(packed_filter);
        free(packed_image);
        free(U);
        free(V);
        free(M);
        free(Y);

        return;
    }

    
    else{

        //计算批次数量
        int batch_size = ti.tile_in_h * ti.tile_in_w;
        int batch_per_stream = batch_size / stream_count;
        int remainder = batch_size % stream_count;
        //printf("Total batch size: %d\n", batch_size);

        //printf("CUDA\n");
          // 计算所需内存大小
        const long long A_size = vs.num_tiles * vs.ic;  // V矩阵大小
        const long long B_size = us.oc * us.ic;        // U矩阵大小
        const long long C_size = vs.num_tiles * us.oc;  // M矩阵大小
        
        size_t pinned_U_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic ;
        size_t pinned_V_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic ;
        size_t pinned_M_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles ;
        
        size_t d_A_req_size = sizeof(float) * batch_size * A_size ;
        size_t d_B_req_size = sizeof(float) * batch_size * B_size ;
        size_t d_C_req_size = sizeof(float) * batch_size * C_size ;


        float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
        float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
        float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

        float *temp_U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
        float *temp_V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.ic * vs.num_tiles);

        if (!pool_initialized) {

              init_cublas();
              pool_initialized = true;
          
        } 
        
        filter_packing(filter, packed_filter, fs);
        image_packing(image, packed_image, is, ti); 

        bool memory_ok = false;

        memory_ok = 
        ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, pinned_U_req_size, true) &&
        ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, pinned_V_req_size, true) &&
        ensure_memory_size((void**)&g_pinned_M, &g_pinned_M_size, pinned_M_req_size, true) &&
        ensure_memory_size((void**)&g_d_A, &g_d_A_size, d_A_req_size, false) &&
        ensure_memory_size((void**)&g_d_B, &g_d_B_size, d_B_req_size, false) &&
        ensure_memory_size((void**)&g_d_C, &g_d_C_size, d_C_req_size, false);


        filter_transform(packed_filter, g_pinned_U, fs, us, us.oc * us.ic);

        image_transform(packed_image, g_pinned_V, vs, ti, vs.ic * vs.num_tiles);

        //偏移量预处理
        int *offset_U = (int *)malloc(sizeof(int) * stream_count);
        int *offset_V = (int *)malloc(sizeof(int) * stream_count);
        int *offset_M = (int *)malloc(sizeof(int) * stream_count);

        int *offset_d_A = (int *)malloc(sizeof(int) * stream_count);
        int *offset_d_B = (int *)malloc(sizeof(int) * stream_count);
        int *offset_d_C = (int *)malloc(sizeof(int) * stream_count);

       // #pragma omp parallel for schedule(dynamic) num_threads(stream_count)
        for(int i = 0; i < stream_count; i++)
        {
            offset_U[i] = i * batch_per_stream * B_size;
            offset_V[i] = i * batch_per_stream * A_size;
            offset_M[i] = i * batch_per_stream * C_size;   

            offset_d_A[i] = i * batch_per_stream * A_size;
            offset_d_B[i] = i * batch_per_stream * B_size;
            offset_d_C[i] = i * batch_per_stream * C_size;
        }

        // 步长 - 每个矩阵的大小（元素数量）
        long long strideA = A_size;
        long long strideB = B_size;
        long long strideC = C_size;
        
        // 执行批处理矩阵乘法
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        cudaEvent_t stream_start_events[stream_count];

        for (int i = 0; i < stream_count; i++) {
          cudaEventCreate(&stream_start_events[i]);
        }

        for(int i = 0; i < stream_count; i++)
        {

            // 确保在同一流中GEMM等待数据传输完成
            cublasSetStream(cublas_handles[i], g_streams[i]);

            if(i > 0)
            {
                cudaStreamWaitEvent(g_streams[i], stream_start_events[i-1], 0);
            }

            cudaMemcpyAsync(g_d_A + offset_d_A[i], g_pinned_V + offset_V[i], batch_per_stream * A_size * sizeof(float), 
                          cudaMemcpyHostToDevice, g_streams[i]);
            cudaMemcpyAsync(g_d_B + offset_d_B[i], g_pinned_U + offset_U[i], batch_per_stream * B_size * sizeof(float),
                          cudaMemcpyHostToDevice, g_streams[i]);


            // 使用带步长的批处理GEMM (在同一流中执行)
            cublasGemmStridedBatchedEx(
                cublas_handles[i],
                CUBLAS_OP_T,           
                CUBLAS_OP_N,
                m,                     
                n,                     
                k,                     
                &alpha,                
                g_d_A + offset_d_A[i],    
                CUDA_R_32F,           
                k,           
                strideA,               
                g_d_B + offset_d_B[i],    
                CUDA_R_32F,            
                k,                     
                strideB,               
                &beta,                 
                g_d_C + offset_d_C[i],    
                CUDA_R_32F,            
                m,                     
                strideC,               
                batch_per_stream,            
                CUDA_R_32F,            
                CUBLAS_GEMM_DEFAULT   
            );

            cudaEventRecord(stream_start_events[i], g_streams[i]);

            cudaMemcpyAsync(g_pinned_M + offset_M[i], g_d_C + offset_d_C[i], batch_per_stream * C_size * sizeof(float), 
                          cudaMemcpyDeviceToHost, g_streams[i]);
            
            cudaEventRecord(events[i], g_streams[i]);
            cudaEventSynchronize(events[i]);

        }

        
        // 输出处理保持不变
        output_transform(g_pinned_M, Y, ti, us.oc * vs.num_tiles);
        output_unpacking_store(Y, out, os, ti);

        // 释放普通内存
        free(packed_filter);
        free(packed_image);
        free(Y);
        
    }

    calledcount++;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////


void winograd_convolution_pipeline(float *__restrict__ image, const int image_height,
  const int image_width, const int input_channel_num,
  float *__restrict__ filter, const int output_channel_num,
  const int batch_num, float *__restrict__ out) {
    // 基本数据准备保持不变
    const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
    const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
    const out_shape_t os = get_output_shape(is, fs);
    const tiling_info_t ti = get_tiling_info(is, os);
    const U_shape_t us = get_U_shape(fs, ti);
    const V_shape_t vs = get_V_shape(is, ti);

    const int m = vs.num_tiles;  // 输出矩阵的行数
    const int n = us.oc;        // 输出矩阵的列数
    const int k = us.ic;    // 内部维度

    const int cpu_limit = 256 *256 * 512; // CPU限制
    if(m*n*k < cpu_limit)
    {
        //printf("CPU\n");
        float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
        float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
        float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
        float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
        float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
        float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);
          
        filter_packing(filter, packed_filter, fs);
        filter_transform(packed_filter, U, fs, us, us.oc * us.ic);
        
        image_packing(image, packed_image, is, ti);
        image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

      //CUDA Prepare
        #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
        for (int64_t h = 0; h < ti.tile_in_h; ++h) {
            for (int64_t w = 0; w < ti.tile_in_w; ++w) {
                typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][us.ic];
                typedef float(*V_tensor_t)[ti.tile_in_w][vs.num_tiles][vs.ic];
                typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
                U_tensor_t U_tensor = (U_tensor_t)U;
                V_tensor_t V_tensor = (V_tensor_t)V;
                M_tensor_t M_tensor = (M_tensor_t)M;
            
                // 初始化 M_tensor[h][w] 为 0
                //memset(M_tensor[h][w], 0, sizeof(float) * us.oc * vs.num_tiles);
                sgemm(vs.num_tiles,
                  us.oc,
                  us.ic,
                  (float *)(V_tensor[h][w]),
                  (float *)(U_tensor[h][w]),
                  (float *)(M_tensor[h][w]));
    
              }
          
        }
            
        output_transform(M, Y, ti, us.oc * vs.num_tiles);
        output_unpacking_store(Y, out, os, ti);
        
        //destroy_cublas();
      
        free(packed_filter);
        free(packed_image);
        free(U);
        free(V);
        free(M);
        free(Y);

        return;
    }

    
    else{

        //计算批次数量
        int batch_size = ti.tile_in_h * ti.tile_in_w;
        int batch_per_stream = batch_size / stream_count;
        int remainder = batch_size % stream_count;
        //printf("Total batch size: %d\n", batch_size);

        //printf("CUDA\n");
          // 计算所需内存大小
        const long long A_size = vs.num_tiles * vs.ic;  // V矩阵大小
        const long long B_size = us.oc * us.ic;        // U矩阵大小
        const long long C_size = vs.num_tiles * us.oc;  // M矩阵大小
        
        size_t pinned_U_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic ;
        size_t pinned_V_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic ;
        size_t pinned_M_req_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles ;
        
        size_t d_A_req_size = sizeof(float) * batch_size * A_size ;
        size_t d_B_req_size = sizeof(float) * batch_size * B_size ;
        size_t d_C_req_size = sizeof(float) * batch_size * C_size ;


        float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
        float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
        float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

        // 在winograd_convolution函数中
        if (!pool_initialized) {
          // 异步初始化CUBLAS

              init_cublas();
              pool_initialized = true;
          
        } 
        
        filter_packing(filter, packed_filter, fs);
        image_packing(image, packed_image, is, ti);      

        bool memory_ok = false;

        if(1)
        {
              memory_ok = 
              ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, pinned_U_req_size, true) &&
              ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, pinned_V_req_size, true) &&
              ensure_memory_size((void**)&g_pinned_M, &g_pinned_M_size, pinned_M_req_size, true) &&
              ensure_memory_size((void**)&g_d_A, &g_d_A_size, d_A_req_size, false) &&
              ensure_memory_size((void**)&g_d_B, &g_d_B_size, d_B_req_size, false) &&
              ensure_memory_size((void**)&g_d_C, &g_d_C_size, d_C_req_size, false);

        }
      
        //transform第一阶段预处理
        filter_transform_stage1(packed_filter, g_pinned_U, fs, us, us.oc * us.ic);
        image_transform_stage1(packed_image, g_pinned_V, vs, ti, vs.ic * vs.num_tiles);


        //偏移量预处理
        //注：这里streamcount = tile_in_h = tile_in_w，所以用流数目替代行数
        int *offset_U = (int *)malloc(sizeof(int) * stream_count);
        int *offset_V = (int *)malloc(sizeof(int) * stream_count);
        int *offset_M = (int *)malloc(sizeof(int) * stream_count);

        int *offset_d_A = (int *)malloc(sizeof(int) * stream_count);
        int *offset_d_B = (int *)malloc(sizeof(int) * stream_count);
        int *offset_d_C = (int *)malloc(sizeof(int) * stream_count);

        for(int i = 0; i < stream_count; i++)
        {
            offset_U[i] = i * batch_per_stream * B_size;
            offset_V[i] = i * batch_per_stream * A_size;
            offset_M[i] = i * batch_per_stream * C_size;   

            offset_d_A[i] = i * batch_per_stream * A_size;
            offset_d_B[i] = i * batch_per_stream * B_size;
            offset_d_C[i] = i * batch_per_stream * C_size;
        }

        //批处理乘法信息
        // 步长 - 每个矩阵的大小（元素数量）
        long long strideA = A_size;
        long long strideB = B_size;
        long long strideC = C_size;
        
        const float alpha = 1.0f;
        const float beta = 0.0f;

        //事件信息
        cudaEvent_t stream_start_events[stream_count];
        cudaEvent_t transform_complete[stream_count];
        cudaEvent_t compute_complete[stream_count];

        for (int i = 0; i < stream_count; i++) {
          cudaEventCreate(&stream_start_events[i]);
          cudaEventCreate(&transform_complete[i]);
          cudaEventCreate(&compute_complete[i]);
        }

        //第一行预处理
        filter_transform_stage2(g_pinned_U, g_pinned_U, fs, us, us.oc * us.ic, 0);
        image_transform_stage2(g_pinned_V, g_pinned_V, vs, ti, vs.ic * vs.num_tiles, 0);
        //cudaEventRecord(transform_complete[0], g_streams[0]);

       /* for(int i = 1; i < stream_count; i++)
        {
            filter_transform_stage2(g_pinned_U, g_pinned_U, fs, us, us.oc * us.ic, i);
            image_transform_stage2(g_pinned_V, g_pinned_V, vs, ti, vs.ic * vs.num_tiles, i);
            //cudaEventRecord(transform_complete[i], g_streams[i]);
        }*/

        for(int i = 0; i < stream_count; i++)
        {
          
          cublasSetStream(cublas_handles[i], g_streams[i]);
          
          cudaStreamWaitEvent(g_streams[i], transform_complete[i], 0);
                      
          cudaMemcpyAsync(g_d_A + offset_d_A[i], g_pinned_V + offset_V[i], batch_per_stream * A_size * sizeof(float), 
                        cudaMemcpyHostToDevice, g_streams[i]);
          cudaMemcpyAsync(g_d_B + offset_d_B[i], g_pinned_U + offset_U[i], batch_per_stream * B_size * sizeof(float),
                      cudaMemcpyHostToDevice, g_streams[i]);
          
            
          cublasGemmStridedBatchedEx(
              cublas_handles[i],
                CUBLAS_OP_T,           // A转置
                CUBLAS_OP_N,           // B不转置
                m,                     // 矩阵C的行数(vs.num_tiles)
                n,                     // 矩阵C的列数(us.oc)
                k,                     // 内部维度(us.ic)
                &alpha,                // 缩放因子
                g_d_A + offset_d_A[i],    // V矩阵起始地址
                CUDA_R_32F,            // 数据类型:float
                k,           // V矩阵的前导维度
                strideA,               // V矩阵序列的步长
                g_d_B + offset_d_B[i],    // U矩阵起始地址
                CUDA_R_32F,            // 数据类型:float
                k,                     // U矩阵的前导维度
                strideB,               // U矩阵序列的步长
                &beta,                 // 缩放因子
                g_d_C + offset_d_C[i],    // M矩阵起始地址
                CUDA_R_32F,            // 数据类型:float
                m,                     // M矩阵的前导维度
                strideC,               // M矩阵序列的步长
                batch_per_stream,            // 批次数量
                CUDA_R_32F,            // 计算类型:float
                CUBLAS_GEMM_DEFAULT    // 使用默认算法
          );

          //cudaEventRecord(stream_start_events[i], g_streams[i]);

          cudaMemcpyAsync(g_pinned_M + offset_M[i], g_d_C + offset_d_C[i], batch_per_stream * C_size * sizeof(float), 
                      cudaMemcpyDeviceToHost, g_streams[i]);
          cudaEventRecord(compute_complete[i], g_streams[i]);
          //printf("DTOH started\n");

          if(i + 1 < stream_count)
          {
              filter_transform_stage2(g_pinned_U, g_pinned_U, fs, us, us.oc * us.ic, i + 1);
              image_transform_stage2(g_pinned_V, g_pinned_V, vs, ti, vs.ic * vs.num_tiles, i + 1);
              cudaEventRecord(transform_complete[i + 1], g_streams[i + 1]);
          }

          cudaEventSynchronize(compute_complete[i]);
          //printf("Stream %d compute complete\n\n", i);

      }
      
      // 输出处理保持不变
      output_transform(g_pinned_M, Y, ti, us.oc * vs.num_tiles);
      output_unpacking_store(Y, out, os, ti);

      // 释放普通内存
      free(packed_filter);
      free(packed_image);
      free(Y);
    
    }

    calledcount++;

}