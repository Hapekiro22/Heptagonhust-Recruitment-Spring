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

// CUDA流
const int stream_count = 12;
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

// 模拟 init_cublas 函数
bool init_cublas() {
    
    //cudaSetDevice(0);
    //printf("成功设置 CUDA 设备 0\n");;

    if(stream_count > 0) {
        for (int i = 0; i < stream_count; i++) {
            cublasCreate(&cublas_handles[i]);
            cudaStreamCreate(&g_streams[i]);
            cublasSetStream(cublas_handles[i], g_streams[i]);
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

// 优化的资源释放
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

const unsigned long long init_memsize = 6000000000; // 4GB

// GPU内存
static half *g_d_A = nullptr;
static half *g_d_B = nullptr;
static half *g_d_C = nullptr;
static size_t g_d_A_size = 0;
static size_t g_d_B_size = 0;
static size_t g_d_C_size = 0;

// 页锁定内存 - 修改为half类型
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

// 确保内存大小足够，如果不足则重新分配
// 修改ensure_memory_size函数以支持写结合内存选项
bool ensure_memory_size(void **mem, size_t *current_size, size_t required_size, bool is_pinned, bool write_combined = false) {
  if (*current_size >= required_size) {
      return true; // 当前内存足够
  }
  
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
      }
  } else {
      err = cudaMalloc(mem, required_size);
  }
  
  if (err != cudaSuccess) {
      *current_size = 0;
      return false;
  }
  
  *current_size = required_size;
  return true;
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

// 第一部分转换 - 行维度转换
    #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
        for (int64_t w = 0; w < ti.tile_in_w; ++w) {
            // 处理每次16个元素
            const int64_t block_size = (16LL < collapsed_dim_size - idx) ? 16 : (collapsed_dim_size - idx);

            // 设置掩码处理边界情况
            __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

            // 加载常量向量
            __m512 vfour = _mm512_set1_ps(4.0f);
            __m512 vneg_four = _mm512_set1_ps(-4.0f);
            __m512 vneg_five = _mm512_set1_ps(-5.0f);
            __m512 vneg_two = _mm512_set1_ps(-2.0f);
            __m512 vtwo = _mm512_set1_ps(2.0f);
            __m512 vone = _mm512_set1_ps(1.0f);
            __m512 vneg_one = _mm512_set1_ps(-1.0f);

            // 向量化变换计算
            __m512 vt0, vt1, vt2, vt3, vt4, vt5;

            // 加载第一个输入行
            __m512 vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[0][w][idx]);

            // 计算第一个变换输出
            vt0 = _mm512_mul_ps(vfour, vz6);

            // 加载第二个输入行并进行变换
            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[1][w][idx]);

            vt1 = _mm512_mul_ps(vneg_four, vz6);
            vt2 = _mm512_mul_ps(vfour, vz6);
            vt3 = _mm512_mul_ps(vneg_two, vz6);
            vt4 = _mm512_mul_ps(vtwo, vz6);
            vt5 = _mm512_mul_ps(vfour, vz6);

            // 加载第三个输入行并更新变换
            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[2][w][idx]);

            vt0 = _mm512_fmadd_ps(vneg_five, vz6, vt0);
            vt1 = _mm512_fmadd_ps(vneg_four, vz6, vt1);
            vt2 = _mm512_fmadd_ps(vneg_four, vz6, vt2);
            vt3 = _mm512_fmadd_ps(vneg_one, vz6, vt3);
            vt4 = _mm512_fmadd_ps(vneg_one, vz6, vt4);

            // 加载第四个输入行并更新变换
            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[3][w][idx]);

            vt1 = _mm512_fmadd_ps(vone, vz6, vt1);
            vt2 = _mm512_fmadd_ps(vneg_one, vz6, vt2);
            vt3 = _mm512_fmadd_ps(vtwo, vz6, vt3);
            vt4 = _mm512_fmadd_ps(vneg_two, vz6, vt4);
            vt5 = _mm512_fmadd_ps(vneg_five, vz6, vt5);

            // 加载第五个输入行并更新变换
            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[4][w][idx]);

            vt0 = _mm512_add_ps(vt0, vz6);
            vt1 = _mm512_add_ps(vt1, vz6);
            vt2 = _mm512_add_ps(vt2, vz6);
            vt3 = _mm512_add_ps(vt3, vz6);
            vt4 = _mm512_add_ps(vt4, vz6);

            // 加载第六个输入行并更新变换
            vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[5][w][idx]);

            vt5 = _mm512_add_ps(vt5, vz6);

            // 存储变换结果
            _mm512_mask_storeu_ps(&V_tensor[0][w][idx], k_mask, vt0);
            _mm512_mask_storeu_ps(&V_tensor[1][w][idx], k_mask, vt1);
            _mm512_mask_storeu_ps(&V_tensor[2][w][idx], k_mask, vt2);
            _mm512_mask_storeu_ps(&V_tensor[3][w][idx], k_mask, vt3);
            _mm512_mask_storeu_ps(&V_tensor[4][w][idx], k_mask, vt4);
            _mm512_mask_storeu_ps(&V_tensor[5][w][idx], k_mask, vt5);
        }
    }

    // 第二部分转换 - 列维度转换
    #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
        for (int64_t h = 0; h < ti.tile_in_h; ++h) {
            // 处理边界
            const int64_t block_size = 16LL < collapsed_dim_size - idx ? 16 : (collapsed_dim_size - idx);
            __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

            // 加载常量向量
            __m512 vfour = _mm512_set1_ps(4.0f);
            __m512 vneg_four = _mm512_set1_ps(-4.0f);
            __m512 vneg_five = _mm512_set1_ps(-5.0f);
            __m512 vneg_two = _mm512_set1_ps(-2.0f);
            __m512 vtwo = _mm512_set1_ps(2.0f);
            __m512 vone = _mm512_set1_ps(1.0f);
            __m512 vneg_one = _mm512_set1_ps(-1.0f);

            // 向量化变换计算
            __m512 vt0, vt1, vt2, vt3, vt4, vt5;

            // 实现列变换
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

            // 存储变换结果
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

// 第一部分转换 - 行维度
#pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
      for (int64_t w = 0; w < fs.w; ++w) {
          // 处理边界
          const int64_t block_size = 16LL < collapsed_dim_size - idx ? 16 : (collapsed_dim_size - idx);
          __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

          // 加载常量系数
          __m512 vquarter = _mm512_set1_ps(1.0f / 4.0f);
          __m512 vneg_sixth = _mm512_set1_ps(-1.0f / 6.0f);
          __m512 vsixth = _mm512_set1_ps(1.0f / 6.0f);
          __m512 vtwentyFourth = _mm512_set1_ps(1.0f / 24.0f);
          __m512 vtwelfth = _mm512_set1_ps(1.0f / 12.0f);
          __m512 vneg_twelfth = _mm512_set1_ps(-1.0f / 12.0f);

          // 向量化变换计算
          __m512 vt0, vt1, vt2, vt3, vt4, vt5;

          // 加载第一个输入行
          __m512 vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][0][idx]);

          vt0 = _mm512_mul_ps(vquarter, vz6);
          vt1 = _mm512_mul_ps(vneg_sixth, vz6);
          vt2 = _mm512_mul_ps(vneg_sixth, vz6);
          vt3 = _mm512_mul_ps(vtwentyFourth, vz6);
          vt4 = _mm512_mul_ps(vtwentyFourth, vz6);

          // 加载第二个输入行并更新变换
          vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][1][idx]);

          vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
          vt2 = _mm512_fmadd_ps(vsixth, vz6, vt2);
          vt3 = _mm512_fmadd_ps(vtwelfth, vz6, vt3);
          vt4 = _mm512_fmadd_ps(vneg_twelfth, vz6, vt4);

          // 加载第三个输入行并更新变换
          vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][2][idx]);

          vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
          vt2 = _mm512_fmadd_ps(vneg_sixth, vz6, vt2);
          vt3 = _mm512_fmadd_ps(vsixth, vz6, vt3);
          vt4 = _mm512_fmadd_ps(vsixth, vz6, vt4);
          vt5 = vz6;  // 直接设置vt5 = vz6

          // 存储变换结果
          _mm512_mask_storeu_ps(&U_tensor[w][0][idx], k_mask, vt0);
          _mm512_mask_storeu_ps(&U_tensor[w][1][idx], k_mask, vt1);
          _mm512_mask_storeu_ps(&U_tensor[w][2][idx], k_mask, vt2);
          _mm512_mask_storeu_ps(&U_tensor[w][3][idx], k_mask, vt3);
          _mm512_mask_storeu_ps(&U_tensor[w][4][idx], k_mask, vt4);
          _mm512_mask_storeu_ps(&U_tensor[w][5][idx], k_mask, vt5);
      }
}

// 第二部分转换 - 列维度
#pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
      for (int64_t h = 0; h < us.h; ++h) {
            // 处理边界
            const int64_t block_size = 16LL < collapsed_dim_size - idx ? 16 : (collapsed_dim_size - idx);
            __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

            // 加载常量系数
            __m512 vquarter = _mm512_set1_ps(1.0f / 4.0f);
            __m512 vneg_sixth = _mm512_set1_ps(-1.0f / 6.0f);
            __m512 vsixth = _mm512_set1_ps(1.0f / 6.0f);
            __m512 vtwentyFourth = _mm512_set1_ps(1.0f / 24.0f);
            __m512 vtwelfth = _mm512_set1_ps(1.0f / 12.0f);
            __m512 vneg_twelfth = _mm512_set1_ps(-1.0f / 12.0f);

            // 向量化变换计算
            __m512 vt0, vt1, vt2, vt3, vt4, vt5;

            // 实现列变换
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

            // 存储变换结果
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


// 第一部分 - 行变换
  #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
  for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
      for (int64_t w = 0; w < ti.tile_in_w; ++w) {
          // 处理边界
          const int64_t block_size = std::min<int64_t>(16, collapsed_dim_size - idx);
          __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

          // 加载常量向量
          __m512 vneg_one = _mm512_set1_ps(-1.0f);
          __m512 vtwo = _mm512_set1_ps(2.0f);
          __m512 vfour = _mm512_set1_ps(4.0f);
          __m512 veight = _mm512_set1_ps(8.0f);
          __m512 vneg_two = _mm512_set1_ps(-2.0f);
          __m512 vneg_eight = _mm512_set1_ps(-8.0f);

          // 向量化变换计算
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

          // 存储结果
          _mm512_mask_storeu_ps(&Y_tensor[0][w][idx], k_mask, vr0);
          _mm512_mask_storeu_ps(&Y_tensor[1][w][idx], k_mask, vr1);
          _mm512_mask_storeu_ps(&Y_tensor[2][w][idx], k_mask, vr2);
          _mm512_mask_storeu_ps(&Y_tensor[3][w][idx], k_mask, vr3);
      }
  }

// 第二部分 - 列变换
  #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t idx = 0; idx < collapsed_dim_size; idx += 16) {
        for (int64_t h = 0; h < ti.tile_out_h; ++h) {
            // 处理边界
            const int64_t block_size = std::min<int64_t>(16, collapsed_dim_size - idx);
            __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;

            // 加载常量向量
            __m512 vneg_one = _mm512_set1_ps(-1.0f);
            __m512 vtwo = _mm512_set1_ps(2.0f);
            __m512 vfour = _mm512_set1_ps(4.0f);
            __m512 veight = _mm512_set1_ps(8.0f);
            __m512 vneg_two = _mm512_set1_ps(-2.0f);
            __m512 vneg_eight = _mm512_set1_ps(-8.0f);

            // 向量化变换计算
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

            // 存储结果
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////
void image_transform_blocked(float *__restrict__ packed_image,
                           float *__restrict__ V,
                           const V_shape_t vs,
                           const tiling_info_t ti,
                           const int64_t collapsed_dim_size) {
    typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
    typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
    packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
    V_tensor_t V_tensor = (V_tensor_t)V;

    // 定义分块参数 - 根据实际缓存大小调整
    const int64_t BLOCK_H = 2;      // 行方向分块
    const int64_t BLOCK_W = 2;      // 列方向分块
    const int64_t BLOCK_C = 128;    // 通道方向分块

    // 行变换分块处理
    #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t c_block = 0; c_block < collapsed_dim_size; c_block += BLOCK_C) {
        for (int64_t w_block = 0; w_block < ti.tile_in_w; w_block += BLOCK_W) {
            for (int64_t vec_idx = 0; vec_idx < BLOCK_C; vec_idx += 16) {
                // 确定实际块边界
                const int64_t max_c = std::min(c_block + BLOCK_C, collapsed_dim_size);
                const int64_t max_w = std::min(w_block + BLOCK_W, ti.tile_in_w);
                const int64_t c = c_block + vec_idx;
                
                // 检查是否超出边界
                if (c >= max_c) continue;
                
                // AVX-512处理的元素数量
                const int64_t block_size = std::min<int64_t>(16, max_c - c);
                __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;
                
                // 预加载常量向量
                __m512 vfour = _mm512_set1_ps(4.0f);
                __m512 vneg_four = _mm512_set1_ps(-4.0f);
                __m512 vneg_five = _mm512_set1_ps(-5.0f);
                __m512 vneg_two = _mm512_set1_ps(-2.0f);
                __m512 vtwo = _mm512_set1_ps(2.0f);
                __m512 vone = _mm512_set1_ps(1.0f);
                __m512 vneg_one = _mm512_set1_ps(-1.0f);
                
                // 对每个块内的列进行处理
                for (int64_t w = w_block; w < max_w; ++w) {
                    // 预取下一个需要的数据 (提高缓存命中率)
                    if (w + 1 < max_w) {
                        _mm_prefetch(&packed_image_tensor[0][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&packed_image_tensor[1][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&packed_image_tensor[2][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&packed_image_tensor[3][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&packed_image_tensor[4][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&packed_image_tensor[5][w+1][c], _MM_HINT_T0);
                    }
                    
                    // 向量化变换计算
                    __m512 vt0, vt1, vt2, vt3, vt4, vt5;
                    __m512 vz6;
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[0][w][c]);
                    vt0 = _mm512_mul_ps(vfour, vz6);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[1][w][c]);
                    vt1 = _mm512_mul_ps(vneg_four, vz6);
                    vt2 = _mm512_mul_ps(vfour, vz6);
                    vt3 = _mm512_mul_ps(vneg_two, vz6);
                    vt4 = _mm512_mul_ps(vtwo, vz6);
                    vt5 = _mm512_mul_ps(vfour, vz6);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[2][w][c]);
                    vt0 = _mm512_fmadd_ps(vneg_five, vz6, vt0);
                    vt1 = _mm512_fmadd_ps(vneg_four, vz6, vt1);
                    vt2 = _mm512_fmadd_ps(vneg_four, vz6, vt2);
                    vt3 = _mm512_fmadd_ps(vneg_one, vz6, vt3);
                    vt4 = _mm512_fmadd_ps(vneg_one, vz6, vt4);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[3][w][c]);
                    vt1 = _mm512_fmadd_ps(vone, vz6, vt1);
                    vt2 = _mm512_fmadd_ps(vneg_one, vz6, vt2);
                    vt3 = _mm512_fmadd_ps(vtwo, vz6, vt3);
                    vt4 = _mm512_fmadd_ps(vneg_two, vz6, vt4);
                    vt5 = _mm512_fmadd_ps(vneg_five, vz6, vt5);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[4][w][c]);
                    vt0 = _mm512_add_ps(vt0, vz6);
                    vt1 = _mm512_add_ps(vt1, vz6);
                    vt2 = _mm512_add_ps(vt2, vz6);
                    vt3 = _mm512_add_ps(vt3, vz6);
                    vt4 = _mm512_add_ps(vt4, vz6);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_image_tensor[5][w][c]);
                    vt5 = _mm512_add_ps(vt5, vz6);
                    
                    // 存储变换结果
                    _mm512_mask_storeu_ps(&V_tensor[0][w][c], k_mask, vt0);
                    _mm512_mask_storeu_ps(&V_tensor[1][w][c], k_mask, vt1);
                    _mm512_mask_storeu_ps(&V_tensor[2][w][c], k_mask, vt2);
                    _mm512_mask_storeu_ps(&V_tensor[3][w][c], k_mask, vt3);
                    _mm512_mask_storeu_ps(&V_tensor[4][w][c], k_mask, vt4);
                    _mm512_mask_storeu_ps(&V_tensor[5][w][c], k_mask, vt5);
                }
            }
        }
    }

    // 列变换分块处理
    #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t c_block = 0; c_block < collapsed_dim_size; c_block += BLOCK_C) {
        for (int64_t h_block = 0; h_block < ti.tile_in_h; h_block += BLOCK_H) {
            for (int64_t vec_idx = 0; vec_idx < BLOCK_C; vec_idx += 16) {
                // 确定实际块边界
                const int64_t max_c = std::min(c_block + BLOCK_C, collapsed_dim_size);
                const int64_t max_h = std::min(h_block + BLOCK_H, ti.tile_in_h);
                const int64_t c = c_block + vec_idx;
                
                // 检查是否超出边界
                if (c >= max_c) continue;
                
                // AVX-512处理的元素数量
                const int64_t block_size = std::min<int64_t>(16, max_c - c);
                __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;
                
                // 预加载常量向量
                __m512 vfour = _mm512_set1_ps(4.0f);
                __m512 vneg_four = _mm512_set1_ps(-4.0f);
                __m512 vneg_five = _mm512_set1_ps(-5.0f);
                __m512 vneg_two = _mm512_set1_ps(-2.0f);
                __m512 vtwo = _mm512_set1_ps(2.0f);
                __m512 vone = _mm512_set1_ps(1.0f);
                __m512 vneg_one = _mm512_set1_ps(-1.0f);
                
                // 对每个块内的行进行处理
                for (int64_t h = h_block; h < max_h; ++h) {
                    // 预取下一行数据
                    if (h + 1 < max_h) {
                        _mm_prefetch(&V_tensor[h+1][0][c], _MM_HINT_T0);
                        _mm_prefetch(&V_tensor[h+1][1][c], _MM_HINT_T0);
                        _mm_prefetch(&V_tensor[h+1][2][c], _MM_HINT_T0);
                        _mm_prefetch(&V_tensor[h+1][3][c], _MM_HINT_T0);
                        _mm_prefetch(&V_tensor[h+1][4][c], _MM_HINT_T0);
                        _mm_prefetch(&V_tensor[h+1][5][c], _MM_HINT_T0);
                    }
                    
                    // 向量化变换计算
                    __m512 vt0, vt1, vt2, vt3, vt4, vt5;
                    __m512 vz6;
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][0][c]);
                    vt0 = _mm512_mul_ps(vfour, vz6);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][1][c]);
                    vt1 = _mm512_mul_ps(vneg_four, vz6);
                    vt2 = _mm512_mul_ps(vfour, vz6);
                    vt3 = _mm512_mul_ps(vneg_two, vz6);
                    vt4 = _mm512_mul_ps(vtwo, vz6);
                    vt5 = _mm512_mul_ps(vfour, vz6);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][2][c]);
                    vt0 = _mm512_fmadd_ps(vneg_five, vz6, vt0);
                    vt1 = _mm512_fmadd_ps(vneg_four, vz6, vt1);
                    vt2 = _mm512_fmadd_ps(vneg_four, vz6, vt2);
                    vt3 = _mm512_fmadd_ps(vneg_one, vz6, vt3);
                    vt4 = _mm512_fmadd_ps(vneg_one, vz6, vt4);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][3][c]);
                    vt1 = _mm512_fmadd_ps(vone, vz6, vt1);
                    vt2 = _mm512_fmadd_ps(vneg_one, vz6, vt2);
                    vt3 = _mm512_fmadd_ps(vtwo, vz6, vt3);
                    vt4 = _mm512_fmadd_ps(vneg_two, vz6, vt4);
                    vt5 = _mm512_fmadd_ps(vneg_five, vz6, vt5);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][4][c]);
                    vt0 = _mm512_add_ps(vt0, vz6);
                    vt1 = _mm512_add_ps(vt1, vz6);
                    vt2 = _mm512_add_ps(vt2, vz6);
                    vt3 = _mm512_add_ps(vt3, vz6);
                    vt4 = _mm512_add_ps(vt4, vz6);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &V_tensor[h][5][c]);
                    vt5 = _mm512_add_ps(vt5, vz6);
                    
                    // 存储最终结果
                    _mm512_mask_storeu_ps(&V_tensor[h][0][c], k_mask, vt0);
                    _mm512_mask_storeu_ps(&V_tensor[h][1][c], k_mask, vt1);
                    _mm512_mask_storeu_ps(&V_tensor[h][2][c], k_mask, vt2);
                    _mm512_mask_storeu_ps(&V_tensor[h][3][c], k_mask, vt3);
                    _mm512_mask_storeu_ps(&V_tensor[h][4][c], k_mask, vt4);
                    _mm512_mask_storeu_ps(&V_tensor[h][5][c], k_mask, vt5);
                }
            }
        }
    }
}

void filter_transform_blocked(float *__restrict__ packed_filter,
                           float *__restrict__ U,
                           const filter_shape_t fs,
                           const U_shape_t us,
                           const int64_t collapsed_dim_size) {
    typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
    typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
    packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
    U_tensor_t U_tensor = (U_tensor_t)U;

    // 定义分块参数
    const int64_t BLOCK_W = 2;      // 列方向分块
    const int64_t BLOCK_H = 2;      // 行方向分块
    const int64_t BLOCK_C = 128;    // 通道方向分块

    // 行变换分块处理
    #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t c_block = 0; c_block < collapsed_dim_size; c_block += BLOCK_C) {
        for (int64_t w_block = 0; w_block < fs.w; w_block += BLOCK_W) {
            for (int64_t vec_idx = 0; vec_idx < BLOCK_C; vec_idx += 16) {
                // 确定实际块边界
                const int64_t max_c = std::min(c_block + BLOCK_C, collapsed_dim_size);
                const int64_t max_w = std::min(w_block + BLOCK_W, fs.w);
                const int64_t c = c_block + vec_idx;
                
                // 检查是否超出边界
                if (c >= max_c) continue;
                
                // AVX-512处理的元素数量
                const int64_t block_size = std::min<int64_t>(16, max_c - c);
                __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;
                
                // 预加载常量向量
                __m512 vquarter = _mm512_set1_ps(1.0f / 4.0f);
                __m512 vneg_sixth = _mm512_set1_ps(-1.0f / 6.0f);
                __m512 vsixth = _mm512_set1_ps(1.0f / 6.0f);
                __m512 vtwentyFourth = _mm512_set1_ps(1.0f / 24.0f);
                __m512 vtwelfth = _mm512_set1_ps(1.0f / 12.0f);
                __m512 vneg_twelfth = _mm512_set1_ps(-1.0f / 12.0f);
                
                // 对每个块内的列进行处理
                for (int64_t w = w_block; w < max_w; ++w) {
                    // 预取下一列数据
                    if (w + 1 < max_w) {
                        _mm_prefetch(&packed_filter_tensor[w+1][0][c], _MM_HINT_T0);
                        _mm_prefetch(&packed_filter_tensor[w+1][1][c], _MM_HINT_T0);
                        _mm_prefetch(&packed_filter_tensor[w+1][2][c], _MM_HINT_T0);
                    }
                    
                    // 向量化变换计算
                    __m512 vt0, vt1, vt2, vt3, vt4, vt5;
                    __m512 vz6;
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][0][c]);
                    vt0 = _mm512_mul_ps(vquarter, vz6);
                    vt1 = _mm512_mul_ps(vneg_sixth, vz6);
                    vt2 = _mm512_mul_ps(vneg_sixth, vz6);
                    vt3 = _mm512_mul_ps(vtwentyFourth, vz6);
                    vt4 = _mm512_mul_ps(vtwentyFourth, vz6);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][1][c]);
                    vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
                    vt2 = _mm512_fmadd_ps(vsixth, vz6, vt2);
                    vt3 = _mm512_fmadd_ps(vtwelfth, vz6, vt3);
                    vt4 = _mm512_fmadd_ps(vneg_twelfth, vz6, vt4);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &packed_filter_tensor[w][2][c]);
                    vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
                    vt2 = _mm512_fmadd_ps(vneg_sixth, vz6, vt2);
                    vt3 = _mm512_fmadd_ps(vsixth, vz6, vt3);
                    vt4 = _mm512_fmadd_ps(vsixth, vz6, vt4);
                    vt5 = vz6;
                    
                    // 存储变换结果
                    _mm512_mask_storeu_ps(&U_tensor[w][0][c], k_mask, vt0);
                    _mm512_mask_storeu_ps(&U_tensor[w][1][c], k_mask, vt1);
                    _mm512_mask_storeu_ps(&U_tensor[w][2][c], k_mask, vt2);
                    _mm512_mask_storeu_ps(&U_tensor[w][3][c], k_mask, vt3);
                    _mm512_mask_storeu_ps(&U_tensor[w][4][c], k_mask, vt4);
                    _mm512_mask_storeu_ps(&U_tensor[w][5][c], k_mask, vt5);
                }
            }
        }
    }

    // 列变换分块处理
    #pragma omp parallel for collapse(2) schedule(guided) num_threads(threads_max)
    for (int64_t c_block = 0; c_block < collapsed_dim_size; c_block += BLOCK_C) {
        for (int64_t h_block = 0; h_block < us.h; h_block += BLOCK_H) {
            for (int64_t vec_idx = 0; vec_idx < BLOCK_C; vec_idx += 16) {
                // 确定实际块边界
                const int64_t max_c = std::min(c_block + BLOCK_C, collapsed_dim_size);
                const int64_t max_h = std::min(h_block + BLOCK_H, us.h);
                const int64_t c = c_block + vec_idx;
                
                // 检查是否超出边界
                if (c >= max_c) continue;
                
                // AVX-512处理的元素数量
                const int64_t block_size = std::min<int64_t>(16, max_c - c);
                __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;
                
                // 预加载常量向量
                __m512 vquarter = _mm512_set1_ps(1.0f / 4.0f);
                __m512 vneg_sixth = _mm512_set1_ps(-1.0f / 6.0f);
                __m512 vsixth = _mm512_set1_ps(1.0f / 6.0f);
                __m512 vtwentyFourth = _mm512_set1_ps(1.0f / 24.0f);
                __m512 vtwelfth = _mm512_set1_ps(1.0f / 12.0f);
                __m512 vneg_twelfth = _mm512_set1_ps(-1.0f / 12.0f);
                
                // 对每个块内的行进行处理
                for (int64_t h = h_block; h < max_h; ++h) {
                    // 预取下一行数据
                    if (h + 1 < max_h) {
                        _mm_prefetch(&U_tensor[0][h+1][c], _MM_HINT_T0);
                        _mm_prefetch(&U_tensor[1][h+1][c], _MM_HINT_T0);
                        _mm_prefetch(&U_tensor[2][h+1][c], _MM_HINT_T0);
                    }
                    
                    // 向量化变换计算
                    __m512 vt0, vt1, vt2, vt3, vt4, vt5;
                    __m512 vz6;
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &U_tensor[0][h][c]);
                    vt0 = _mm512_mul_ps(vquarter, vz6);
                    vt1 = _mm512_mul_ps(vneg_sixth, vz6);
                    vt2 = _mm512_mul_ps(vneg_sixth, vz6);
                    vt3 = _mm512_mul_ps(vtwentyFourth, vz6);
                    vt4 = _mm512_mul_ps(vtwentyFourth, vz6);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &U_tensor[1][h][c]);
                    vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
                    vt2 = _mm512_fmadd_ps(vsixth, vz6, vt2);
                    vt3 = _mm512_fmadd_ps(vtwelfth, vz6, vt3);
                    vt4 = _mm512_fmadd_ps(vneg_twelfth, vz6, vt4);
                    
                    vz6 = _mm512_maskz_loadu_ps(k_mask, &U_tensor[2][h][c]);
                    vt1 = _mm512_fmadd_ps(vneg_sixth, vz6, vt1);
                    vt2 = _mm512_fmadd_ps(vneg_sixth, vz6, vt2);
                    vt3 = _mm512_fmadd_ps(vsixth, vz6, vt3);
                    vt4 = _mm512_fmadd_ps(vsixth, vz6, vt4);
                    vt5 = vz6;
                    
                    // 存储最终结果
                    _mm512_mask_storeu_ps(&U_tensor[0][h][c], k_mask, vt0);
                    _mm512_mask_storeu_ps(&U_tensor[1][h][c], k_mask, vt1);
                    _mm512_mask_storeu_ps(&U_tensor[2][h][c], k_mask, vt2);
                    _mm512_mask_storeu_ps(&U_tensor[3][h][c], k_mask, vt3);
                    _mm512_mask_storeu_ps(&U_tensor[4][h][c], k_mask, vt4);
                    _mm512_mask_storeu_ps(&U_tensor[5][h][c], k_mask, vt5);
                }
            }
        }
    }
}


void output_transform_blocked(float *__restrict__ M,
                           float *__restrict__ Y,
                           const tiling_info_t ti,
                           const int64_t collapsed_dim_size) {
    typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
    typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
    M_tensor_t M_tensor = (M_tensor_t)M;
    Y_tensor_t Y_tensor = (Y_tensor_t)Y;

    // 定义分块参数
    const int64_t BLOCK_H = 2;      // 行方向分块
    const int64_t BLOCK_W = 2;      // 列方向分块
    const int64_t BLOCK_C = 128;    // 通道方向分块

    // 行变换分块处理
    #pragma omp parallel for collapse(3) schedule(guided) num_threads(threads_max)
    for (int64_t c_block = 0; c_block < collapsed_dim_size; c_block += BLOCK_C) {
        for (int64_t w_block = 0; w_block < ti.tile_in_w; w_block += BLOCK_W) {
            for (int64_t vec_idx = 0; vec_idx < BLOCK_C; vec_idx += 16) {
                // 确定实际块边界
                const int64_t max_c = std::min(c_block + BLOCK_C, collapsed_dim_size);
                const int64_t max_w = std::min(w_block + BLOCK_W, ti.tile_in_w);
                const int64_t c = c_block + vec_idx;
                
                // 检查是否超出边界
                if (c >= max_c) continue;
                
                // AVX-512处理的元素数量
                const int64_t block_size = std::min<int64_t>(16, max_c - c);
                __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;
                
                // 预加载常量向量
                __m512 vneg_one = _mm512_set1_ps(-1.0f);
                __m512 vtwo = _mm512_set1_ps(2.0f);
                __m512 vfour = _mm512_set1_ps(4.0f);
                __m512 veight = _mm512_set1_ps(8.0f);
                __m512 vneg_two = _mm512_set1_ps(-2.0f);
                __m512 vneg_eight = _mm512_set1_ps(-8.0f);
                
                // 对每个块内的列进行处理
                for (int64_t w = w_block; w < max_w; ++w) {
                    // 预取下一列数据
                    if (w + 1 < max_w) {
                        _mm_prefetch(&M_tensor[0][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&M_tensor[1][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&M_tensor[2][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&M_tensor[3][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&M_tensor[4][w+1][c], _MM_HINT_T0);
                        _mm_prefetch(&M_tensor[5][w+1][c], _MM_HINT_T0);
                    }
                    
                    // 向量化变换计算
                    __m512 vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[0][w][c]);
                    __m512 vr0 = vz4;
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[1][w][c]);
                    vr0 = _mm512_add_ps(vr0, vz4);
                    __m512 vr1 = vz4;
                    __m512 vr2 = vz4;
                    __m512 vr3 = vz4;
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[2][w][c]);
                    vr0 = _mm512_add_ps(vr0, vz4);
                    vr1 = _mm512_fmadd_ps(vneg_one, vz4, vr1);
                    vr2 = _mm512_add_ps(vr2, vz4);
                    vr3 = _mm512_fmadd_ps(vneg_one, vz4, vr3);
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[3][w][c]);
                    vr0 = _mm512_add_ps(vr0, vz4);
                    vr1 = _mm512_fmadd_ps(vtwo, vz4, vr1);
                    vr2 = _mm512_fmadd_ps(vfour, vz4, vr2);
                    vr3 = _mm512_fmadd_ps(veight, vz4, vr3);
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[4][w][c]);
                    vr0 = _mm512_add_ps(vr0, vz4);
                    vr1 = _mm512_fmadd_ps(vneg_two, vz4, vr1);
                    vr2 = _mm512_fmadd_ps(vfour, vz4, vr2);
                    vr3 = _mm512_fmadd_ps(vneg_eight, vz4, vr3);
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &M_tensor[5][w][c]);
                    vr3 = _mm512_add_ps(vr3, vz4);
                    
                    // 存储结果
                    _mm512_mask_storeu_ps(&Y_tensor[0][w][c], k_mask, vr0);
                    _mm512_mask_storeu_ps(&Y_tensor[1][w][c], k_mask, vr1);
                    _mm512_mask_storeu_ps(&Y_tensor[2][w][c], k_mask, vr2);
                    _mm512_mask_storeu_ps(&Y_tensor[3][w][c], k_mask, vr3);
                }
            }
        }
    }

    // 列变换分块处理
    #pragma omp parallel for collapse(3) schedule(guided) num_threads(threads_max)
    for (int64_t c_block = 0; c_block < collapsed_dim_size; c_block += BLOCK_C) {
        for (int64_t h_block = 0; h_block < ti.tile_out_h; h_block += BLOCK_H) {
            for (int64_t vec_idx = 0; vec_idx < BLOCK_C; vec_idx += 16) {
                // 确定实际块边界
                const int64_t max_c = std::min(c_block + BLOCK_C, collapsed_dim_size);
                const int64_t max_h = std::min(h_block + BLOCK_H, ti.tile_out_h);
                const int64_t c = c_block + vec_idx;
                
                // 检查是否超出边界
                if (c >= max_c) continue;
                
                // AVX-512处理的元素数量
                const int64_t block_size = std::min<int64_t>(16, max_c - c);
                __mmask16 k_mask = (block_size == 16) ? 0xFFFF : (1 << block_size) - 1;
                
                // 预加载常量向量
                __m512 vneg_one = _mm512_set1_ps(-1.0f);
                __m512 vtwo = _mm512_set1_ps(2.0f);
                __m512 vfour = _mm512_set1_ps(4.0f);
                __m512 veight = _mm512_set1_ps(8.0f);
                __m512 vneg_two = _mm512_set1_ps(-2.0f);
                __m512 vneg_eight = _mm512_set1_ps(-8.0f);
                
                // 对每个块内的行进行处理
                for (int64_t h = h_block; h < max_h; ++h) {
                    // 预取下一行数据
                    if (h + 1 < max_h) {
                        _mm_prefetch(&Y_tensor[h+1][0][c], _MM_HINT_T0);
                        _mm_prefetch(&Y_tensor[h+1][1][c], _MM_HINT_T0);
                        _mm_prefetch(&Y_tensor[h+1][2][c], _MM_HINT_T0);
                        _mm_prefetch(&Y_tensor[h+1][3][c], _MM_HINT_T0);
                        _mm_prefetch(&Y_tensor[h+1][4][c], _MM_HINT_T0);
                        _mm_prefetch(&Y_tensor[h+1][5][c], _MM_HINT_T0);
                    }
                    
                    // 向量化变换计算
                    __m512 vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][0][c]);
                    __m512 vr0 = vz4;
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][1][c]);
                    vr0 = _mm512_add_ps(vr0, vz4);
                    __m512 vr1 = vz4;
                    __m512 vr2 = vz4;
                    __m512 vr3 = vz4;
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][2][c]);
                    vr0 = _mm512_add_ps(vr0, vz4);
                    vr1 = _mm512_fmadd_ps(vneg_one, vz4, vr1);
                    vr2 = _mm512_add_ps(vr2, vz4);
                    vr3 = _mm512_fmadd_ps(vneg_one, vz4, vr3);
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][3][c]);
                    vr0 = _mm512_add_ps(vr0, vz4);
                    vr1 = _mm512_fmadd_ps(vtwo, vz4, vr1);
                    vr2 = _mm512_fmadd_ps(vfour, vz4, vr2);
                    vr3 = _mm512_fmadd_ps(veight, vz4, vr3);
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][4][c]);
                    vr0 = _mm512_add_ps(vr0, vz4);
                    vr1 = _mm512_fmadd_ps(vneg_two, vz4, vr1);
                    vr2 = _mm512_fmadd_ps(vfour, vz4, vr2);
                    vr3 = _mm512_fmadd_ps(vneg_eight, vz4, vr3);
                    
                    vz4 = _mm512_maskz_loadu_ps(k_mask, &Y_tensor[h][5][c]);
                    vr3 = _mm512_add_ps(vr3, vz4);
                    
                    // 存储最终结果
                    _mm512_mask_storeu_ps(&Y_tensor[h][0][c], k_mask, vr0);
                    _mm512_mask_storeu_ps(&Y_tensor[h][1][c], k_mask, vr1);
                    _mm512_mask_storeu_ps(&Y_tensor[h][2][c], k_mask, vr2);
                    _mm512_mask_storeu_ps(&Y_tensor[h][3][c], k_mask, vr3);
                }
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////

void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

  #pragma omp parallel for collapse(2) schedule(guided)
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

  #pragma omp parallel for collapse(2) schedule(guided)
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

  #pragma omp parallel for collapse(4) schedule(guided) num_threads(threads_max)
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

  // 定义分块大小
  const int BM = 32; // M方向分块
  const int BN = 32; // N方向分块
  const int BK = 32; // K方向分块

  //#pragma omp parallel for schedule(guided) collapse(2) num_threads(threads_max)
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
            
            // 水平求和
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



void winograd_convolution_singlestream(float *__restrict__ image, const int image_height,
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


    
    // CPU代码保持不变...
    else {
    // 计算所需内存大小
    const long long A_size = vs.num_tiles * vs.ic;  // V矩阵大小
    const long long B_size = us.oc * us.ic;        // U矩阵大小
    const long long C_size = vs.num_tiles * us.oc;  // M矩阵大小
    
    size_t pinned_U_req_size = sizeof(float) * batch_size * B_size;
    size_t pinned_V_req_size = sizeof(float) * batch_size * A_size;
    size_t pinned_M_req_size = sizeof(float) * batch_size * C_size;
    
    size_t d_A_req_size = sizeof(float) * batch_size * A_size;
    size_t d_B_req_size = sizeof(float) * batch_size * B_size;
    size_t d_C_req_size = sizeof(float) * batch_size * C_size;
    
    // 初始化 cuBLAS 与 CUDA 流
    if(!init_flag) {
        init_cublas();
        init_flag = true;
    }
    
    if (!pool_initialized) {
        // 创建CUDA流
        if (g_stream == NULL && cudaStreamCreate(&g_stream) != cudaSuccess) {
            fprintf(stderr, "Failed to create CUDA stream\n");
            return;
        }
        
        // 预分配GPU内存（如果配置了预分配）
        if(mem_pre_allocated) {
           ensure_memory_size((void**)&g_d_A, &g_d_A_size, init_memsize, false);
           ensure_memory_size((void**)&g_d_B, &g_d_B_size, init_memsize, false);
           ensure_memory_size((void**)&g_d_C, &g_d_C_size, init_memsize, false);
        }
        
        pool_initialized = true;
    }
    
    // 确保GPU内存足够
    bool memory_ok = 
      ensure_memory_size((void**)&g_d_A, &g_d_A_size, d_A_req_size, false) &&
      ensure_memory_size((void**)&g_d_B, &g_d_B_size, d_B_req_size, false) &&
      ensure_memory_size((void**)&g_d_C, &g_d_C_size, d_C_req_size, false);
    
    // 设置 cublas 流
    cublasSetStream(cublas_handle, g_stream);

    // 分配普通堆内存用于计算
    float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
    float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
    float *temp_U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
    float *temp_V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.ic * vs.num_tiles);
    float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

    // 使用普通堆内存进行变换计算
    filter_packing(filter, packed_filter, fs);
    filter_transform(packed_filter, temp_U, fs, us, us.oc * us.ic);

    image_packing(image, packed_image, is, ti);
    image_transform(packed_image, temp_V, vs, ti, vs.ic * vs.num_tiles);

    // 计算完成后，仅为传输分配写结合内存
    bool wc_memory_ok = 
      ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, pinned_U_req_size, true, true) &&
      ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, pinned_V_req_size, true, true);
    
    // 分配接收数据的普通锁页内存
    ensure_memory_size((void**)&g_pinned_M, &g_pinned_M_size, pinned_M_req_size, true, false);

    if (!wc_memory_ok) {
      fprintf(stderr, "Failed to allocate write-combining memory, falling back to regular pinned memory\n");
      // 分配普通锁页内存作为回退
      ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, pinned_U_req_size, true, false);
      ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, pinned_V_req_size, true, false);
    }

    // 直接从堆内存复制到已分配的锁页内存(普通或写结合)
    memcpy(g_pinned_V, temp_V, batch_size * A_size * sizeof(float));
    memcpy(g_pinned_U, temp_U, batch_size * B_size * sizeof(float));

    // 释放临时计算内存
    free(temp_U);
    free(temp_V);
    free(packed_filter);
    free(packed_image);

    // 从锁页内存传输到GPU
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
        CUBLAS_OP_T,           // A转置
        CUBLAS_OP_N,           // B不转置
        m,                     // 矩阵C的行数(vs.num_tiles)
        n,                     // 矩阵C的列数(us.oc)
        k,                     // 内部维度(us.ic)
        &alpha,                // 缩放因子
        g_d_A,                 // V矩阵起始地址
        CUDA_R_32F,            // 数据类型:float
        k,                     // V矩阵的前导维度
        strideA,               // V矩阵序列的步长
        g_d_B,                 // U矩阵起始地址
        CUDA_R_32F,            // 数据类型:float
        k,                     // U矩阵的前导维度
        strideB,               // U矩阵序列的步长
        &beta,                 // 缩放因子
        g_d_C,                 // M矩阵起始地址
        CUDA_R_32F,            // 数据类型:float
        m,                     // M矩阵的前导维度
        strideC,               // M矩阵序列的步长
        batch_size,            // 批次数量
        CUDA_R_32F,            // 计算类型:float
        CUBLAS_GEMM_DEFAULT    // 使用默认算法
    );

    // 异步将结果复制回主机
    cudaMemcpyAsync(g_pinned_M, g_d_C, batch_size * C_size * sizeof(float), 
                  cudaMemcpyDeviceToHost, g_stream);
    
    // 在流上同步，确保传输完成
    cudaStreamSynchronize(g_stream);
    
    // 输出处理
    output_transform(g_pinned_M, Y, ti, us.oc * vs.num_tiles);
    output_unpacking_store(Y, out, os, ti);

    // 释放剩余内存
    free(Y);
  }

}


//------------------------------------------------Finish--------------------------------------------------//

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

    const int m = vs.num_tiles;  // 输出矩阵的行数
    const int n = us.oc;        // 输出矩阵的列数
    const int k = us.ic;    // 内部维度


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
          filter_transform_blocked(packed_filter, U, fs, us, us.oc * us.ic);
        
          image_packing(image, packed_image, is, ti);
          image_transform_blocked(packed_image, V, vs, ti, vs.ic * vs.num_tiles);
      
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
            
        output_transform_blocked(M, Y, ti, us.oc * vs.num_tiles);
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
        
        // 初始化 cuBLAS (如果还没有初始化)
        /*if(!init_flag) {
            init_cublas();
            init_flag = true;
        }*/


        // 初始化内存池（如果是第一次使用）
        if (!pool_initialized) {
            // 创建CUDA流            
            
            init_cublas();exit(0);
            
             // 预分配所有内存，确保大小足够
             if(mem_pre_allocated)
             {
                //分配总内存，然后平分给每个矩阵
             }
            
            // 标记为已初始化
            pool_initialized = true;
        }
        


        float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
        float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
        float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

        // 使用普通堆内存进行变换计算
        float *temp_U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
        float *temp_V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.ic * vs.num_tiles);

        bool memory_ok = false;

        //ensure时间很长，避免这时cpu空闲
        //#pragma omp parallel sections
        {
            //#pragma omp section
            {        
                filter_packing(filter, packed_filter, fs);
                filter_transform(packed_filter, temp_U, fs, us, us.oc * us.ic);
                image_packing(image, packed_image, is, ti);
                image_transform(packed_image, temp_V, vs, ti, vs.ic * vs.num_tiles);
            }

            //#pragma omp section
            {         
              // 确保所有内存大小足够
                 memory_ok = 
                    ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, pinned_U_req_size, true) &&
                    ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, pinned_V_req_size, true) &&
                    ensure_memory_size((void**)&g_pinned_M, &g_pinned_M_size, pinned_M_req_size, true) &&
                    ensure_memory_size((void**)&g_d_A, &g_d_A_size, d_A_req_size, false) &&
                    ensure_memory_size((void**)&g_d_B, &g_d_B_size, d_B_req_size, false) &&
                    ensure_memory_size((void**)&g_d_C, &g_d_C_size, d_C_req_size, false);
            }
        }

        if(memory_ok == false)
        {
            fprintf(stderr,"Failed to allocate memory\n");
            free(packed_filter);
            free(packed_image);
            free(temp_U);
            free(temp_V);
            return;
        }
        memcpy(g_pinned_V, temp_V, batch_size * A_size * sizeof(float));
        memcpy(g_pinned_U, temp_U, batch_size * B_size * sizeof(float));

        // 普通内存分配（这些较小，可以每次重新分配）

       // filter_packing(filter, packed_filter, fs);
       // filter_transform(packed_filter, g_pinned_U, fs, us, us.oc * us.ic);
       // image_packing(image, packed_image, is, ti);
       // image_transform(packed_image, g_pinned_V, vs, ti, vs.ic * vs.num_tiles);


        //分批次处理
        //#pragma omp parallel for schedule(dynamic) num_threads(stream_count*2)
        for(int i = 0; i < stream_count; i++)
        {

            int stream_batch_count = batch_per_stream + ((i < remainder) ? 1 : 0);
            int stream_batch_start = i * batch_per_stream;
           // printf("Stream %d: Processing %d batches (start: %d)\n", i, stream_batch_count, stream_batch_start);


            //计算偏移量
            size_t offset_U = stream_batch_start * B_size;
            size_t offset_V = stream_batch_start * A_size;
            size_t offset_M = stream_batch_start * C_size;   

            size_t offset_d_A = stream_batch_start * A_size;
            size_t offset_d_B = stream_batch_start * B_size;
            size_t offset_d_C = stream_batch_start * C_size;

            //计算剩余批次数量
            //printf("Batches remaining: %d\n", batch_size - (i + 1) * batch_per_stream);

            // 使用异步内存复制将数据传输到GPU
            cudaMemcpyAsync(g_d_A + offset_d_A, g_pinned_V + offset_V, stream_batch_count * A_size * sizeof(float), 
                          cudaMemcpyHostToDevice, g_streams[i]);
            cudaMemcpyAsync(g_d_B + offset_d_B, g_pinned_U + offset_U, stream_batch_count * B_size * sizeof(float),
                          cudaMemcpyHostToDevice, g_streams[i]);


            // 步长 - 每个矩阵的大小（元素数量）
            long long strideA = A_size;
            long long strideB = B_size;
            long long strideC = C_size; 

            // 执行批处理矩阵乘法
            const float alpha = 1.0f;
            const float beta = 0.0f;


            //用事件同步流
           // cudaEventRecord(events[i], g_streams[i]);

             // 确保在同一流中GEMM等待数据传输完成
            cublasSetStream(cublas_handles[i], g_streams[i]);

            // 使用带步长的批处理GEMM (在同一流中执行)
            cublasGemmStridedBatchedEx(
                cublas_handles[i],
                CUBLAS_OP_T,           // A转置
                CUBLAS_OP_N,           // B不转置
                m,                     // 矩阵C的行数(vs.num_tiles)
                n,                     // 矩阵C的列数(us.oc)
                k,                     // 内部维度(us.ic)
                &alpha,                // 缩放因子
                g_d_A + offset_d_A,    // V矩阵起始地址
                CUDA_R_32F,            // 数据类型:float
                k,           // V矩阵的前导维度
                strideA,               // V矩阵序列的步长
                g_d_B + offset_d_B,    // U矩阵起始地址
                CUDA_R_32F,            // 数据类型:float
                k,                     // U矩阵的前导维度
                strideB,               // U矩阵序列的步长
                &beta,                 // 缩放因子
                g_d_C + offset_d_C,    // M矩阵起始地址
                CUDA_R_32F,            // 数据类型:float
                m,                     // M矩阵的前导维度
                strideC,               // M矩阵序列的步长
                stream_batch_count,            // 批次数量
                CUDA_R_32F,            // 计算类型:float
                CUBLAS_GEMM_DEFAULT    // 使用默认算法
            );

            //if()
            //cudaStreamSynchronize(g_streams[i]);
            

            // 异步将结果复制回主机（使用页锁定内存）
            cudaMemcpyAsync(g_pinned_M + offset_M, g_d_C + offset_d_C, stream_batch_count * C_size * sizeof(float), 
                          cudaMemcpyDeviceToHost, g_streams[i]);
            
            cudaEventRecord(events[i], g_streams[i]);
            cudaEventSynchronize(events[i]);

        }

        // 等待所有流完成
        //#pragma omp parallel for schedule(static) num_threads(stream_count)
        for(int i = 0; i < stream_count; i++)
        {
            //cudaEventSynchronize(events[i]);
           // cudaStreamSynchronize(g_streams[i]);
        }

        //cudaDeviceSynchronize();
        
        // 输出处理保持不变
        output_transform(g_pinned_M, Y, ti, us.oc * vs.num_tiles);
        output_unpacking_store(Y, out, os, ti);

        // 释放普通内存
        free(packed_filter);
        free(packed_image);
        free(Y);
        
    }

}

void winograd_convolution_fulled(float *__restrict__ image, const int image_height,
  const int image_width, const int input_channel_num,
  float *__restrict__ filter, const int output_channel_num,
  const int batch_num, float *__restrict__ out) {
    
    // 基本数据准备
    const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
    const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
    const out_shape_t os = get_output_shape(is, fs);
    const tiling_info_t ti = get_tiling_info(is, os);
    const U_shape_t us = get_U_shape(fs, ti);
    const V_shape_t vs = get_V_shape(is, ti);

    const int m = vs.num_tiles;  // 输出矩阵的行数
    const int n = us.oc;        // 输出矩阵的列数
    const int k = us.ic;        // 内部维度

    const int cpu_limit = 256 * 256 * 512; // CPU限制
    if(m*n*k < cpu_limit) {
        // 小规模问题使用CPU计算
        winograd_convolution(image, image_height, image_width, 
                                   input_channel_num, filter, 
                                   output_channel_num, batch_num, out);
        return;
    }

    // 计算批次数量
    int batch_size = ti.tile_in_h * ti.tile_in_w;
    
    // 动态确定最优流数量，不超过总批次数和最大流数量
    int optimal_streams = stream_count;
    if (optimal_streams > batch_size) {
        optimal_streams = batch_size;
    }
    
    if (optimal_streams == 0) {
        optimal_streams = 1; // 至少使用一个流
    }
    
    // 计算每个流处理的批次数
    int batch_per_stream = batch_size / optimal_streams;
    int remainder = batch_size % optimal_streams;
    
    // 创建事件数组，用于流之间的同步
    cudaEvent_t upload_events[optimal_streams];
    cudaEvent_t compute_events[optimal_streams]; 
    cudaEvent_t download_events[optimal_streams];
    
    // 创建事件并关闭计时以减少开销
    for (int i = 0; i < optimal_streams; i++) {
        cudaEventCreateWithFlags(&upload_events[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&compute_events[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&download_events[i], cudaEventDisableTiming);
    }

    // 计算矩阵大小
    const long long A_size = vs.num_tiles * vs.ic;  // V矩阵大小
    const long long B_size = us.oc * us.ic;        // U矩阵大小
    const long long C_size = vs.num_tiles * us.oc;  // M矩阵大小
    
    // 内存需求计算
    size_t pinned_U_req_size = sizeof(float) * batch_size * B_size;
    size_t pinned_V_req_size = sizeof(float) * batch_size * A_size;
    size_t pinned_M_req_size = sizeof(float) * batch_size * C_size;
    
    size_t d_A_req_size = sizeof(float) * batch_size * A_size;
    size_t d_B_req_size = sizeof(float) * batch_size * B_size;
    size_t d_C_req_size = sizeof(float) * batch_size * C_size;
    
    // 初始化内存池（如果需要）
    if (!pool_initialized) {
        init_cublas();
        
        if(mem_pre_allocated) {

        }
        
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
    
    if (!memory_ok) {
        //fprintf(stderr, "内存分配失败，回退到CPU实现\n");
        winograd_convolution(image, image_height, image_width, 
                                   input_channel_num, filter, 
                                   output_channel_num, batch_num, out);
        return;
    }
    
    // 普通内存分配
    float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
    float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
    float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

    // 数据转换
    filter_packing(filter, packed_filter, fs);
    filter_transform(packed_filter, g_pinned_U, fs, us, us.oc * us.ic);

    image_packing(image, packed_image, is, ti);
    image_transform(packed_image, g_pinned_V, vs, ti, vs.ic * vs.num_tiles);

    // 步长配置
    long long strideA = A_size;
    long long strideB = B_size;
    long long strideC = C_size;
    
    // GEMM参数
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 计算每个流的处理范围
    int batch_starts[optimal_streams];
    int batch_counts[optimal_streams];
    
    // 正确计算每个流处理的批次范围
    int current_batch = 0;
    for (int i = 0; i < optimal_streams; i++) {
        batch_starts[i] = current_batch;
        batch_counts[i] = batch_per_stream + (i < remainder ? 1 : 0);
        current_batch += batch_counts[i];
    }

    // ===== 第一阶段: 所有上传任务 =====
    for(int i = 0; i < optimal_streams; i++) {
        // 计算偏移量 (元素数)
        size_t offset_U = batch_starts[i] * B_size;
        size_t offset_V = batch_starts[i] * A_size;
        size_t offset_d_A = offset_V;  // d_A对应V
        size_t offset_d_B = offset_U;  // d_B对应U
        
        // 异步上传数据，不等待
        cudaMemcpyAsync(g_d_A + offset_d_A, g_pinned_V + offset_V, 
                      batch_counts[i] * A_size * sizeof(float), 
                      cudaMemcpyHostToDevice, g_streams[i]);
        cudaMemcpyAsync(g_d_B + offset_d_B, g_pinned_U + offset_U, 
                      batch_counts[i] * B_size * sizeof(float),
                      cudaMemcpyHostToDevice, g_streams[i]);
                      
        // 记录上传完成事件
        cudaEventRecord(upload_events[i], g_streams[i]);
    }

    // ===== 第二阶段: 所有计算任务 =====
    for(int i = 0; i < optimal_streams; i++) {
        // 计算偏移量 (元素数)
        size_t offset_V = batch_starts[i] * A_size;
        size_t offset_U = batch_starts[i] * B_size;
        size_t offset_M = batch_starts[i] * C_size;
        
        size_t offset_d_A = offset_V;  // d_A对应V
        size_t offset_d_B = offset_U;  // d_B对应U
        size_t offset_d_C = offset_M;  // d_C对应M
        
        // 设置计算前的依赖关系 - 关键步骤: 等待自己的上传完成
        cudaStreamWaitEvent(g_streams[i], upload_events[i], 0);
        
        // 设置流
        cublasSetStream(cublas_handles[i], g_streams[i]);
        
        // 执行GEMM计算
        cublasGemmStridedBatchedEx(
            cublas_handles[i],
            CUBLAS_OP_T,           // A转置
            CUBLAS_OP_N,           // B不转置
            m,                     // 矩阵C的行数
            n,                     // 矩阵C的列数
            k,                     // 内部维度
            &alpha,                // 缩放因子
            g_d_A + offset_d_A,    // V矩阵
            CUDA_R_32F,            // 数据类型
            k,                     // V矩阵前导维度 (使用和单流版本相同的设置)
            strideA,               // V矩阵步长
            g_d_B + offset_d_B,    // U矩阵
            CUDA_R_32F,            // 数据类型
            k,                     // U矩阵前导维度 (使用和单流版本相同的设置)
            strideB,               // U矩阵步长
            &beta,                 // 缩放因子
            g_d_C + offset_d_C,    // M矩阵
            CUDA_R_32F,            // 数据类型
            m,                     // M矩阵前导维度
            strideC,               // M矩阵步长
            batch_counts[i],       // 批次数量
            CUDA_R_32F,            // 计算类型
            CUBLAS_GEMM_DEFAULT    // 算法选择
        );
        
        // 记录计算完成事件
        cudaEventRecord(compute_events[i], g_streams[i]);
    }

    // ===== 第三阶段: 所有下载任务 =====
    for(int i = 0; i < optimal_streams; i++) {
        // 计算偏移量 (元素数)
        size_t offset_M = batch_starts[i] * C_size;
        size_t offset_d_C = offset_M;  // d_C对应M
        
        // 设置下载前的依赖关系 - 关键步骤: 等待自己的计算完成
        cudaStreamWaitEvent(g_streams[i], compute_events[i], 0);
        
        // 异步下载结果
        cudaMemcpyAsync(g_pinned_M + offset_M, g_d_C + offset_d_C,
                      batch_counts[i] * C_size * sizeof(float),
                      cudaMemcpyDeviceToHost, g_streams[i]);
                      
        // 记录下载完成事件
        cudaEventRecord(download_events[i], g_streams[i]);
    }

    // ===== 等待所有流完成 =====
    for(int i = 0; i < optimal_streams; i++) {
        // 确保所有下载都完成了
        cudaEventSynchronize(download_events[i]);
    }
    
    // 添加全局同步点，确保所有GPU操作完成
    cudaDeviceSynchronize();
    
    // 输出转换
    output_transform(g_pinned_M, Y, ti, us.oc * vs.num_tiles);
    output_unpacking_store(Y, out, os, ti);

    // 释放内存和事件
    free(packed_filter);
    free(packed_image);
    free(Y);
    
    for (int i = 0; i < optimal_streams; i++) {
        cudaEventDestroy(upload_events[i]);
        cudaEventDestroy(compute_events[i]);
        cudaEventDestroy(download_events[i]);
    }
}



//-----------------------------------------------Finished--------------------------------------------------// 
//---------------------------------------------------------------------------------------------------------//
//-----------------------------------------------Finished--------------------------------------------------//






void showMatrix(float *matrix, int pre) {

    for(int i = 0; i < pre; i++)
    {
        for(int j = 0; j < pre; j++)
        {
            printf("%f ", matrix[i * pre + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void showMatrix_h(__nv_bfloat16 *matrix, int pre) {

    for(int i = 0; i < pre; i++)
    {
        for(int j = 0; j < pre; j++)
        {
          float val = __bfloat162float(matrix[i * pre + j]);
          printf("%f ", val); // 正确转换为float后打印
        }
        printf("\n");
    }
    printf("\n");
}

bool test_bf16_support() {
  __nv_bfloat16 test_val = __float2bfloat16(1.5f);
  float back = __bfloat162float(test_val);
  printf("BF16测试: 1.5 -> BF16 -> %f\n", back);
  
  float test_values[] = {-210.0f, -40.0f, 40.0f, 210.0f, 0.062f, 0.125f};
  for (int i = 0; i < 6; i++) {
      __nv_bfloat16 bf_val = __float2bfloat16(test_values[i]);
      float back = __bfloat162float(bf_val);
      printf("原值: %f, 转换后: %f\n", test_values[i], back);
  }
  
  return fabs(back - 1.5f) < 0.1f;
}

void winograd_convolution_banned(float *__restrict__ image, const int image_height,
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

        // 添加测试代码
        static bool bf16_tested = false;
        if (!bf16_tested) {
            bf16_tested = true;
            bool bf16_works = test_bf16_support();
            printf("BF16支持测试: %s\n", bf16_works ? "通过" : "失败");
        }

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


    
    else {
      //printf("CUDA with FP16\n");
      // 计算所需内存大小
      const long long A_size = vs.num_tiles * vs.ic;  // V矩阵大小
      const long long B_size = us.oc * us.ic;        // U矩阵大小
      const long long C_size = vs.num_tiles * us.oc;  // M矩阵大小

      size_t pinned_U_req_size = sizeof(float) * batch_size * B_size;
      size_t pinned_V_req_size = sizeof(float) * batch_size * A_size;
      size_t pinned_M_req_size = sizeof(float) * batch_size * C_size;

      size_t pinned_U_bf16_req_size = sizeof(__nv_bfloat16) * batch_size * B_size;
      size_t pinned_V_bf16_req_size = sizeof(__nv_bfloat16) * batch_size * A_size;
      size_t pinned_M_bf16_req_size = sizeof(__nv_bfloat16) * batch_size * C_size;
      
      size_t d_A_bf16_req_size = sizeof(__nv_bfloat16) * batch_size * A_size;
      size_t d_B_bf16_req_size = sizeof(__nv_bfloat16) * batch_size * B_size;
      size_t d_C_bf16_req_size = sizeof(__nv_bfloat16) * batch_size * C_size;


       // 确保所有内存大小足够
       bool memory_ok = 
            ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, pinned_U_req_size, true) &&
            ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, pinned_V_req_size, true) &&
            ensure_memory_size((void**)&g_pinned_M, &g_pinned_M_size, pinned_M_req_size, true) &&
            ensure_memory_size((void**)&g_pinned_U_bf16, &g_pinned_U_bf16_size, pinned_U_bf16_req_size, true) &&
            ensure_memory_size((void**)&g_pinned_V_bf16, &g_pinned_V_bf16_size, pinned_V_bf16_req_size, true) &&
            ensure_memory_size((void**)&g_pinned_M_bf16, &g_pinned_M_bf16_size, pinned_M_bf16_req_size, true) &&
            ensure_memory_size((void**)&g_d_A_bf16, &g_d_A_bf16_size, d_A_bf16_req_size, false) &&
            ensure_memory_size((void**)&g_d_B_bf16, &g_d_B_bf16_size, d_B_bf16_req_size, false) &&
            ensure_memory_size((void**)&g_d_C_bf16, &g_d_C_bf16_size, d_C_bf16_req_size, false);
            
      // 普通内存分配（这些较小，可以每次重新分配）
      float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
      float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
      float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

      // 准备数据阶段保持不变
      filter_packing(filter, packed_filter, fs);
      filter_transform(packed_filter, g_pinned_U, fs, us, us.oc * us.ic);

      image_packing(image, packed_image, is, ti);
      image_transform(packed_image, g_pinned_V, vs, ti, vs.ic * vs.num_tiles);

      //printf("Before conversion:\n");
      //showMatrix(g_pinned_V, 4);
      //showMatrix(g_pinned_U, 4);


      // 转换FP32-->BF16
      //#pragma omp parallel for
      for (int64_t i = 0; i < batch_size * A_size; i++) {
          g_pinned_V_bf16[i] = __float2bfloat16(g_pinned_V[i]);
      }
      
      //#pragma omp parallel for
      for (int64_t i = 0; i < batch_size * B_size; i++) {
          g_pinned_U_bf16[i] = __float2bfloat16(g_pinned_U[i]);
      }

      printf("After conversion:\n");
      showMatrix_h(g_pinned_V_bf16, 4);
      showMatrix_h(g_pinned_U_bf16, 4);

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
             if(0)
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
      
      
      // 设置 cublas 流
      cublasSetStream(cublas_handle, g_stream);

        // 将BF16数据复制到GPU
        cudaMemcpyAsync(g_d_A_bf16, g_pinned_V_bf16, batch_size * A_size * sizeof(__nv_bfloat16), 
                      cudaMemcpyHostToDevice, g_stream);
        cudaMemcpyAsync(g_d_B_bf16, g_pinned_U_bf16, batch_size * B_size * sizeof(__nv_bfloat16), 
                      cudaMemcpyHostToDevice, g_stream);

        // 步长 - 每个矩阵的大小（元素数量）
        long long strideA = A_size;
        long long strideB = B_size;
        long long strideC = C_size;

        // 执行批处理矩阵乘法
        const __nv_bfloat16 alpha_bf16 = __float2bfloat16(1.0f);
        const __nv_bfloat16 beta_bf16 = __float2bfloat16(0.0f);

//////////////////////////////////////////////////////////////////////////////////////////////////
      // 在 GEMM 调用前添加测试矩阵
{
  // 创建简单的 2x2 矩阵乘法测试
  __nv_bfloat16 test_A[4] = {
      __float2bfloat16(1.0f), __float2bfloat16(2.0f),
      __float2bfloat16(3.0f), __float2bfloat16(4.0f)
  };
  __nv_bfloat16 test_B[4] = {
      __float2bfloat16(5.0f), __float2bfloat16(6.0f),
      __float2bfloat16(7.0f), __float2bfloat16(8.0f)
  };
  __nv_bfloat16 test_C[4] = {
      __float2bfloat16(0.0f), __float2bfloat16(0.0f),
      __float2bfloat16(0.0f), __float2bfloat16(0.0f)
  };
  
  __nv_bfloat16 *d_test_A, *d_test_B, *d_test_C;
  cudaMalloc(&d_test_A, 4 * sizeof(__nv_bfloat16));
  cudaMalloc(&d_test_B, 4 * sizeof(__nv_bfloat16));
  cudaMalloc(&d_test_C, 4 * sizeof(__nv_bfloat16));
  
  cudaMemcpy(d_test_A, test_A, 4 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_B, test_B, 4 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_C, test_C, 4 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  
  __nv_bfloat16 alpha_bf16_test = __float2bfloat16(1.0f);
  __nv_bfloat16 beta_bf16_test = __float2bfloat16(0.0f);
  
  // 测试简单的矩阵乘法
  cublasGemmEx(
      cublas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      2, 2, 2,
      &alpha_bf16_test,
      d_test_A, CUDA_R_16BF, 2,
      d_test_B, CUDA_R_16BF, 2,
      &beta_bf16_test,
      d_test_C, CUDA_R_16BF, 2,
      CUDA_R_32F,
      CUBLAS_GEMM_ALGO0
  );
  
  __nv_bfloat16 result_C[4];
  cudaMemcpy(result_C, d_test_C, 4 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  printf("测试矩阵乘法结果:\n");
  printf("%f %f\n", __bfloat162float(result_C[0]), __bfloat162float(result_C[1]));
  printf("%f %f\n", __bfloat162float(result_C[2]), __bfloat162float(result_C[3]));
  printf("\n\n");
  
  cudaFree(d_test_A);
  cudaFree(d_test_B);
  cudaFree(d_test_C);
}

////////////////////////////////////////////////////////////////////////////////////////////////

        // 使用带步长的批处理GEMM (在同一流中执行)
      cublasGemmStridedBatchedEx(
          cublas_handle,
          CUBLAS_OP_T,     // A转置
          CUBLAS_OP_N,     // B不转置
          m,               // 矩阵C的行数(vs.num_tiles)
          n,               // 矩阵C的列数(us.oc)
          k,               // 内部维度(us.ic)
          &alpha_bf16,          // 缩放因子
          g_d_A_bf16,           // V矩阵起始地址
          CUDA_R_16BF,      // 数据类型:float
          k,               // V矩阵的前导维度
          strideA,         // V矩阵序列的步长
          g_d_B_bf16,           // U矩阵起始地址
          CUDA_R_16BF,      // 数据类型:float
          k,               // U矩阵的前导维度
          strideB,         // U矩阵序列的步长
          &beta_bf16,           // 缩放因子
          g_d_C_bf16,           // M矩阵起始地址
          CUDA_R_16BF,      // 数据类型:float
          m,               // M矩阵的前导维度
          strideC,         // M矩阵序列的步长
          batch_size,      // 批次数量
          CUBLAS_COMPUTE_32F,  // 计算类型 (cublasComputeType_t)
          CUBLAS_GEMM_DEFAULT
      );

        // 复制结果回主机
        cudaMemcpyAsync(g_pinned_M_bf16, g_d_C_bf16, batch_size * C_size * sizeof(__nv_bfloat16), 
                      cudaMemcpyDeviceToHost, g_stream);
        
        
        // 在流上同步，确保传输完成
        cudaStreamSynchronize(g_stream);

        printf("After conversion:\n");
        showMatrix_h(g_pinned_M_bf16, 4);

        // 转换BF16-->FP32
        #pragma omp parallel for
        for(int64_t i = 0; i < batch_size * C_size; i++) {
            g_pinned_M[i] = __bfloat162float(g_pinned_M_bf16[i]);
        }

        
        // 输出处理保持不变
        output_transform(g_pinned_M, Y, ti, us.oc * vs.num_tiles);
        output_unpacking_store(Y, out, os, ti);

        free(packed_filter);
        free(packed_image);
        free(Y);
        
    }

}