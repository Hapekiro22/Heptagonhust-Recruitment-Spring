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

//callback
// DTOH回调所需的数据结构
struct DtohCallbackData {
  cudaStream_t next_stream;  // 下一个需要唤醒的流
  int stream_index;          // 当前流索引（用于调试）
  cudaEvent_t dtoh_begin_event; // DTOH开始事件
};

// 回调函数数组
static DtohCallbackData* g_callback_data[stream_count] = {nullptr};

// DTOH开始时的回调函数
void CUDART_CB dtohBeginCallback(cudaStream_t stream, cudaError_t status, void* userData) {
  DtohCallbackData* data = (DtohCallbackData*)userData;
  
  // 记录DTOH开始事件
  cudaEventRecord(data->dtoh_begin_event, stream);
  
  // 使下一个流等待此事件
  if (data->next_stream != nullptr) {
      cudaStreamWaitEvent(data->next_stream, data->dtoh_begin_event, 0);
  }
  
  #ifdef DEBUG_CALLBACK
  printf("DTOH Begin Callback triggered for stream %d\n", data->stream_index);
  #endif
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


//-------------------------------CUDA define-------------------------------------//

//-------------------------------MEMORY define-------------------------------------//


// 内存池全局变量
static bool pool_initialized = false;

static bool mem_pre_allocated = false;

const unsigned long long init_memsize = 6000000000; // 4GB

// GPU内存
static float *g_d_A = nullptr;
static float *g_d_B = nullptr;
static float *g_d_C = nullptr;
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
  //printf("current size: %zu, required size: %zu\n", *current_size, required_size);
  if (*current_size >= required_size ) {
      return true; // 当前内存足够
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

//bf16 convert
__global__ void convertFP32ToBF16Kernel(float* input, __nv_bfloat16* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
      output[idx] = __float2bfloat16(input[idx]);
  }
}

// CPU端函数，将BF16转回FP32
void convertBF16ToFP32_fast(const __nv_bfloat16* input, float* output, size_t size) {
  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < size; i++) {
      // 将BF16视为uint16_t，然后移位成uint32_t
      uint16_t bf16_bits = *reinterpret_cast<const uint16_t*>(&input[i]);
      uint32_t fp32_bits = static_cast<uint32_t>(bf16_bits) << 16;
      memcpy(&output[i], &fp32_bits, sizeof(float));
  }
}

void convertBF16ToFP32(const __nv_bfloat16* input, float* output, size_t size) {
  constexpr size_t BLOCK_SIZE = 1024; // L1缓存友好的大小
  
  #pragma omp parallel
  {
      #pragma omp for schedule(guided)
      for (size_t block = 0; block < size; block += BLOCK_SIZE) {
          size_t end = std::min(block + BLOCK_SIZE, size);
          
          // 处理当前块的所有元素
          for (size_t i = block; i < end; i++) {
              // 将BF16视为uint16_t
              uint16_t bf16_bits = *reinterpret_cast<const uint16_t*>(&input[i]);
              uint32_t fp32_bits = static_cast<uint32_t>(bf16_bits) << 16;
              memcpy(&output[i], &fp32_bits, sizeof(float));
          }
      }
  }
}
///////////////////////////////////






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

/////////////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////////

void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

  #pragma omp parallel for collapse(4) schedule(guided) //num_threads(fs.h*fs.w)
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

  #pragma omp parallel for collapse(4) schedule(guided) //num_threads(ti.num_tiles * is.ic)
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

  #pragma omp parallel for collapse(4) schedule(guided) //num_threads(ti.tile_out_h * ti.tile_out_w)
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

    //printf("The input float size is %lld\n", sizeof(float) * is.h * is.w * is.ic);
    //printf("The filter float size is %lld\n", sizeof(float) * fs.h * fs.w * fs.ic * fs.oc);

    const int cpu_limit = 512 * 256 * 256; // CPU限制
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
        filter_transform_256(packed_filter, U, fs, us, us.oc * us.ic);
        
        image_packing(image, packed_image, is, ti);
        image_transform_256(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

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
            
        output_transform_256(M, Y, ti, us.oc * vs.num_tiles);
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
        size_t pinned_M_bf16_req_size = sizeof(__nv_bfloat16) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles ;
        
        size_t d_A_req_size = sizeof(float) * batch_size * A_size ;
        size_t d_B_req_size = sizeof(float) * batch_size * B_size ;
        size_t d_C_req_size = sizeof(float) * batch_size * C_size ;
        size_t d_C_bf16_req_size = sizeof(__nv_bfloat16) * batch_size * C_size ;


        // 初始化内存池（如果是第一次使用）
        if (!pool_initialized) {
            // 创建CUDA流            
            //printf("Creating CUDA stream\n");
            init_cublas();
            //exit(0);
            
            // 标记为已初始化
            pool_initialized = true;
        }
        


        float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
        float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
        float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);
        float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);

        // 使用普通堆内存进行变换计算
        float *temp_U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
        float *temp_V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.ic * vs.num_tiles);

        bool memory_ok = false;

        if(1)
        {
              memory_ok = 
              ensure_memory_size((void**)&g_pinned_U, &g_pinned_U_size, pinned_U_req_size, true) &&
              ensure_memory_size((void**)&g_pinned_V, &g_pinned_V_size, pinned_V_req_size, true) &&
              ensure_memory_size((void**)&g_pinned_M_bf16, &g_pinned_M_bf16_size, pinned_M_bf16_req_size, true) &&
              ensure_memory_size((void**)&g_d_A, &g_d_A_size, d_A_req_size, false) &&
              ensure_memory_size((void**)&g_d_B, &g_d_B_size, d_B_req_size, false) &&
              ensure_memory_size((void**)&g_d_C, &g_d_C_size, d_C_req_size, false) &&
              ensure_memory_size((void**)&g_d_C_bf16, &g_d_C_bf16_size, d_C_bf16_req_size, false);

        }

        // 普通内存分配（这些较小，可以每次重新分配）
        filter_packing(filter, packed_filter, fs);
        filter_transform_256(packed_filter, g_pinned_U, fs, us, us.oc * us.ic);
        image_packing(image, packed_image, is, ti);
        image_transform_256(packed_image, g_pinned_V, vs, ti, vs.ic * vs.num_tiles);
        //printf("Filter and image transformed\n");

        //各个流延迟时间
        float delay_between_streams = 2.4f;

        //偏移量预处理
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
          //cudaEventCreate(&stream_end_events[i]);
        }

        //分批次处理
        //#pragma omp parallel for schedule(dynamic) num_threads(stream_count*2)
        for(int i = 0; i < stream_count; i++)
        {

            // 确保在同一流中GEMM等待数据传输完成
            cublasSetStream(cublas_handles[i], g_streams[i]);

            if(i > 0)
            {
                cudaStreamWaitEvent(g_streams[i], stream_start_events[i-1], 0);
            }


            // 使用异步内存复制将数据传输到GPU
            cudaMemcpyAsync(g_d_A + offset_d_A[i], g_pinned_V + offset_V[i], batch_per_stream * A_size * sizeof(float), 
                          cudaMemcpyHostToDevice, g_streams[i]);
            cudaMemcpyAsync(g_d_B + offset_d_B[i], g_pinned_U + offset_U[i], batch_per_stream * B_size * sizeof(float),
                          cudaMemcpyHostToDevice, g_streams[i]);

            // 使用带步长的批处理GEMM (在同一流中执行)
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

            ////////////////////
            // 计算内核网格和块大小
            int blockSize = 256;
            int numBlocks = (batch_per_stream * C_size + blockSize - 1) / blockSize;
            
            // 关键步骤：转换FP32结果为BF16
            convertFP32ToBF16Kernel<<<numBlocks, blockSize, 0, g_streams[i]>>>(
                g_d_C + offset_d_C[i],          // FP32输入
                g_d_C_bf16 + offset_d_C[i],     // BF16输出
                batch_per_stream * C_size
            );
            
            ////////////////////////

            // 异步将结果复制回主机（使用页锁定内存）
            cudaMemcpyAsync(g_pinned_M_bf16 + offset_M[i], g_d_C_bf16 + offset_d_C[i], batch_per_stream * C_size * sizeof(__nv_bfloat16), 
                          cudaMemcpyDeviceToHost, g_streams[i]);
            
            cudaEventRecord(events[i], g_streams[i]);
            cudaEventSynchronize(events[i]);

            convertBF16ToFP32(
              g_pinned_M_bf16 + offset_M[i],
              M + offset_M[i],
              batch_per_stream * C_size
          );

        }

        // 等待所有流完成
        //#pragma omp parallel for schedule(static) num_threads(stream_count)
       for(int i = 0; i < stream_count; i++)
        {
            cudaEventSynchronize(events[i]);
            //cudaStreamSynchronize(g_streams[i]);
        }

        
        // 输出处理保持不变
        output_transform(M, Y, ti, us.oc * vs.num_tiles);
        output_unpacking_store(Y, out, os, ti);

        // 释放普通内存
        free(packed_filter);
        free(packed_image);
        free(M);
        free(Y);
        
    }

    calledcount++;

}