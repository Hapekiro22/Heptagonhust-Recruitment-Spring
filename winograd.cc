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

//-------------------------------CUDA define-------------------------------------//

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define DEBUGs 1

bool use_gpu = true;
bool init_flag = false;

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
    
    //cudaSetDevice(0);
    //printf("成功设置 CUDA 设备 0\n");
    
    cublasCreate(&cublas_handle);
    //printf("成功创建 cuBLAS 句柄\n");

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

void sgemm_cublas_batched(const int64_t M, const int64_t N, const int64_t K, 
  float *U_data, float *V_data, float *M_data,
  const tiling_info_t ti) {
      const int batch_size = ti.tile_in_h * ti.tile_in_w;

      // 准备设备内存指针数组
      float **h_A_array = (float**)malloc(batch_size * sizeof(float*));
      float **h_B_array = (float**)malloc(batch_size * sizeof(float*));
      float **h_C_array = (float**)malloc(batch_size * sizeof(float*));

      float **d_A_array, **d_B_array, **d_C_array;
      float *d_A, *d_B, *d_C;

      // A、B、C 的大小（以元素为单位）
      const int A_size = K * M;  // V 矩阵: vs.ic * vs.num_tiles
      const int B_size = K * N;  // U 矩阵: us.ic * us.oc
      const int C_size = M * N;  // M 矩阵: vs.num_tiles * us.oc

      // 分配设备内存
      cudaMalloc((void**)&d_A, batch_size * A_size * sizeof(float));
      cudaMalloc((void**)&d_B, batch_size * B_size * sizeof(float));
      cudaMalloc((void**)&d_C, batch_size * C_size * sizeof(float));
      cudaMalloc((void**)&d_A_array, batch_size * sizeof(float*));
      cudaMalloc((void**)&d_B_array, batch_size * sizeof(float*));
      cudaMalloc((void**)&d_C_array, batch_size * sizeof(float*));

      // 复制数据到设备并设置指针数组
      for (int idx = 0; idx < batch_size; idx++) {
      int h = idx / ti.tile_in_w;
      int w = idx % ti.tile_in_w;

      // 计算V, U, M在大数组中的偏移
      float *V_ptr = V_data + idx * A_size;
      float *U_ptr = U_data + idx * B_size;
      float *M_ptr = M_data + idx * C_size;

      // 将V, U, M复制到设备内存
      cudaMemcpy(d_A + idx * A_size, V_ptr, A_size * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B + idx * B_size, U_ptr, B_size * sizeof(float), cudaMemcpyHostToDevice);

      // 设置指针数组
      h_A_array[idx] = d_A + idx * A_size;
      h_B_array[idx] = d_B + idx * B_size;
      h_C_array[idx] = d_C + idx * C_size;
      }

      // 将指针数组复制到设备
      cudaMemcpy(d_A_array, h_A_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B_array, h_B_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice);
      cudaMemcpy(d_C_array, h_C_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice);

      // 执行批处理矩阵乘法
      const float alpha = 1.0f;
      const float beta = 0.0f;

      cublasSgemmBatched(cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      M,  // vs.num_tiles
      N,  // us.oc
      K,  // us.ic
      &alpha,
      (const float**)d_A_array,
      K,  // leading dimension of A
      (const float**)d_B_array,
      K,  // leading dimension of B
      &beta,
      d_C_array,
      M,  // leading dimension of C
      batch_size);

      // 将结果复制回主机
      for (int idx = 0; idx < batch_size; idx++) {
      int h = idx / ti.tile_in_w;
      int w = idx % ti.tile_in_w;
      float *M_ptr = M_data + idx * C_size;

      cudaMemcpy(M_ptr, d_C + idx * C_size, C_size * sizeof(float), cudaMemcpyDeviceToHost);
      }

      // 释放资源
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
      cudaFree(d_A_array);
      cudaFree(d_B_array);
      cudaFree(d_C_array);
      free(h_A_array);
      free(h_B_array);
      free(h_C_array);
}

//-------------------------------CUDA define-------------------------------------//



struct alignas(64) parameters {
  float z0, z1, z2, z3, z4, z5, z6, z7; //内存对齐
};


void image_transform(float *__restrict__ packed_image,
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


void filter_transform(float *__restrict__ packed_filter,
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

void output_transform(float *__restrict__ M,
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

  #pragma omp parallel for schedule(guided) collapse(2) num_threads(threads_max)
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



void winograd_convolution_single(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
  const int image_height,
  const int image_width,
  const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
  const int output_channel_num,
  const int batch_num,
  float *__restrict__ out) {
  /* new vars of shape */
  const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
  const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
  const out_shape_t os = get_output_shape(is, fs);
  const tiling_info_t ti = get_tiling_info(is, os);
  const U_shape_t us = get_U_shape(fs, ti);
  const V_shape_t vs = get_V_shape(is, ti);
  
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

  const long ccpu_limit = 64*64*64;

  bool use_cuda = init_cublas();
  
  //#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(threads_max)
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
      
        // 根据是否使用 CUDA 选择实现
          if(vs.num_tiles * us.ic * us.oc < ccpu_limit) {
            sgemm(vs.num_tiles,
              us.oc,
              us.ic,
              (float *)(V_tensor[h][w]),
              (float *)(U_tensor[h][w]),
              (float *)(M_tensor[h][w]));

          } 
          else {
            // CUDA 实现
              sgemm_cublas(vs.num_tiles,
                  us.oc,
                  us.ic,
                  (float *)(V_tensor[h][w]),
                  (float *)(U_tensor[h][w]),
                  (float *)(M_tensor[h][w]));
            }
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

    // 分配和初始化内存
    float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
    float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
    float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
    float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
    float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
    float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

    // 准备数据阶段保持不变
    filter_packing(filter, packed_filter, fs);
    filter_transform(packed_filter, U, fs, us, us.oc * us.ic);

    image_packing(image, packed_image, is, ti);
    image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

    int batch_size = ti.tile_in_h * ti.tile_in_w;

    if(batch_size < 2048)
    {
        
    }

    // 初始化 cuBLAS (如果还没有初始化)
    if(!init_flag)
    {
        init_cublas();
        init_flag = true;
    }

    

    // 准备连续内存数据
    // 创建连续数据缓冲区


    // 分配GPU内存
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    const int m = vs.num_tiles;  // 输出矩阵的行数
    const int n = us.oc;        // 输出矩阵的列数
    const int k = us.ic;        // 内部维度

    // 计算每个矩阵的大小
    const long long A_size = vs.num_tiles * vs.ic;  // V矩阵大小
    const long long B_size = us.oc * us.ic;        // U矩阵大小
    const long long C_size = vs.num_tiles * us.oc;  // M矩阵大小

    // 分配GPU内存
    cudaMalloc((void**)&d_A, batch_size * A_size * sizeof(float));
    cudaMalloc((void**)&d_B, batch_size * B_size * sizeof(float));
    cudaMalloc((void**)&d_C, batch_size * C_size * sizeof(float));

    // 将数据复制到GPU
    cudaMemcpy(d_A, V, batch_size * A_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, U, batch_size * B_size * sizeof(float), cudaMemcpyHostToDevice);

    // 设置前导维度和步长
    int lda = k;        // V矩阵的前导维度
    int ldb = k;        // U矩阵的前导维度
    int ldc = m;        // M矩阵的前导维度

    // 步长 - 每个矩阵的大小（元素数量）
    long long strideA = A_size;
    long long strideB = B_size;
    long long strideC = C_size;

    // 执行批处理矩阵乘法
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 使用带步长的批处理GEMM
    cublasGemmStridedBatchedEx(
    cublas_handle,
    CUBLAS_OP_T,     // A转置
    CUBLAS_OP_N,     // B不转置
    m,               // 矩阵C的行数(vs.num_tiles)
    n,               // 矩阵C的列数(us.oc)
    k,               // 内部维度(us.ic)
    &alpha,          // 缩放因子
    d_A,             // V矩阵起始地址
    CUDA_R_32F,      // 数据类型:float
    k,               // V矩阵的前导维度
    strideA,         // V矩阵序列的步长
    d_B,             // U矩阵起始地址
    CUDA_R_32F,      // 数据类型:float
    k,               // U矩阵的前导维度
    strideB,         // U矩阵序列的步长
    &beta,           // 缩放因子
    d_C,             // M矩阵起始地址
    CUDA_R_32F,      // 数据类型:float
    m,               // M矩阵的前导维度
    strideC,         // M矩阵序列的步长
    batch_size,      // 批次数量
    CUDA_R_32F,      // 计算类型:float
    CUBLAS_GEMM_DEFAULT // 使用默认算法
    );

    // 将结果复制回主机
    float *h_C_data = (float *)malloc(batch_size * C_size * sizeof(float));
    cudaMemcpy(M, d_C, batch_size * C_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    /*// 将结果重新组织回原始格式
    #pragma omp parallel for schedule(guided) num_threads(threads_max)
    for (int i = 0; i < batch_size; i++) {
    int h = i / ti.tile_in_w;
    int w = i % ti.tile_in_w;

    typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
    M_tensor_t M_tensor = (M_tensor_t)M;

    size_t c_offset = i * C_size;
    memcpy(M_tensor[h][w], h_C_data + c_offset, C_size * sizeof(float));
    }*/

    // 释放GPU内存
    //cudaFree(d_A);
    //cudaFree(d_B);
    //cudaFree(d_C);

    // 释放主机内存
   // free(h_A_data);
    //free(h_B_data);
    //free(h_C_data);

    // cuBLAS资源清理
    //destroy_cublas();

    // 输出处理保持不变
    output_transform(M, Y, ti, us.oc * vs.num_tiles);
    output_unpacking_store(Y, out, os, ti);

    // 释放资源
    free(packed_filter);
    free(packed_image);
    free(U);
    free(V);
    free(M);
    free(Y);
}