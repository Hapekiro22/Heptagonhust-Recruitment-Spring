#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "utils.h"

//-----------------------------------------------CUDA------------------------------------------------------//

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

bool init = false;

//Error check
#define CHECK_CUDA_ERROR(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
  } \
}

#define CHECK_CUBLAS_ERROR(call) { \
  cublasStatus_t status = call; \
  if (status != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS error in %s at line %d: %d\n", __FILE__, __LINE__, status); \
      exit(EXIT_FAILURE); \
  } \
}

//gpu init
bool initCUDA() {
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable device found\n");
        return false;
    }
    
    int device = 0;
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device));
    
    printf("Using CUDA device: %s\n", deviceProp.name);
    CHECK_CUDA_ERROR(cudaSetDevice(device));
    
    return true;
}

cublasHandle_t cublas_handle;
cudaStream_t stream;

//-------------------------------------------------CUDA----------------------------------------------------//

//--------------------------------------------memory management--------------------------------------------//

static bool pool_initialized = false;

static bool mem_pre_allocated = false;
const unsigned long long init_mem_size = 1024 * 1024 * 1024; // 1GB

//gpu内存
static float *g_image = nullptr;
static float *g_filter = nullptr;
static float *g_output = nullptr;
static size_t gimage_cur_size = 0;
static size_t gfilter_cur_size = 0;
static size_t goutput_cur_size = 0;

//页锁定内存
static float *pin_image = nullptr;
static float *pin_filter = nullptr;
static float *pin_output = nullptr;
static size_t pimage_cur_size = 0;
static size_t pfilter_cur_size = 0;
static size_t poutput_cur_size = 0;

//
static float *g_packed_image = nullptr;
static float *g_packed_filter = nullptr;
static float *g_U = nullptr;
static float *g_V = nullptr;
static float *g_M = nullptr;
static float *g_Y = nullptr;

// 对应的大小变量
static size_t g_packed_image_size = 0;
static size_t g_packed_filter_size = 0;
static size_t g_U_size = 0;
static size_t g_V_size = 0;
static size_t g_M_size = 0;
static size_t g_Y_size = 0;


bool ensure_mem_size(void **mem, size_t *current_size, size_t req_size, bool pinned)
{

    if(*current_size >= req_size) {
        return true;
    }

    if(*mem)
    {
        if(pinned){
            cudaFreeHost(*mem);
        }
        else{
            cudaFree(*mem);
        }
    }

    if(pinned){
      cudaMallocHost(mem, req_size);
    }
    else{
      cudaMalloc(mem, req_size);
    }    

    *current_size = req_size;
    return true;
    
}



//--------------------------------------------memory management--------------------------------------------//



//--------------------------------------image_processing------------------------------------//
__global__ void image_packing_kernel(
      float *__restrict__ image,
      float *__restrict__ packed_image,
      const image_shape_t is,
      const tilling_info_t ti)
{

      int tile = blockIdx.x * blockDim.x + threadIdx.x;
      int ic = blockIdx.y * blockDim.y + threadIdx.y;

      if(tile >= ti.num_tiles || ic >= is.ic) return;

      tile_index_t tidx = get_tile_index(tile,ti);
      int batch = tidx.b;
      int hh = tidx.th;
      int ww = tidx.tw;

      const int64_t image_batch_offset = batch * is.ic * is.h * is.w;
      const int64_t image_channel_offset = ic * is.h * is.w;
      
      for (int h = 0; h < ti.tile_in_h; ++h) {
          for (int w = 0; w < ti.tile_in_w; ++w) {
              // 计算源索引和目标索引
              int64_t src_h = hh * 4 + h;
              int64_t src_w = ww * 4 + w;
              
              // 计算目标位置
              int64_t dst_idx = ((h * ti.tile_in_w + w) * ti.num_tiles + tile) * is.ic + ic;
              
              // 检查边界并赋值
              if (src_h < is.h && src_w < is.w) {
                  int64_t src_idx = image_batch_offset + image_channel_offset + src_h * is.w + src_w;
                  packed_image[dst_idx] = image[src_idx];
              } else {
                  packed_image[dst_idx] = 0.0f;
              }
          }
      }

}

__global__ void filter_packing_kernel(
  float *__restrict__ filter,
  float *__restrict__ packed_filter,
  const filter_shape_t fs)
{
  // 计算线程索引
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int w = blockIdx.y * blockDim.y + threadIdx.y;
  int oc = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (h >= fs.h || w >= fs.w || oc >= fs.oc) return;
  
  // 获取filter_tensor和packed_filter_tensor的对应关系
  int64_t packed_filter_offset = (h * fs.w + w) * fs.oc * fs.ic + oc * fs.ic;
  int64_t filter_oc_offset = oc * fs.ic * fs.h * fs.w;
  
  for (int ic = 0; ic < fs.ic; ++ic) {
      int64_t filter_idx = filter_oc_offset + ic * fs.h * fs.w + h * fs.w + w;
      packed_filter[packed_filter_offset + ic] = filter[filter_idx];
  }
}


__global__ void  image_transform_kernel(
  float *packed_image,          // CHWN 布局的打包输入
  float *V,                     // 输出变换结果
  const tiling_info_t ti,       // 分块信息
  const int64_t num_tiles,      // 块数量
  const int64_t ic)
{

      //每个线程块处理32*8个输入块
      const int tid = threadIdx.x;
      const int batch_idx = blockIdx.x * 32;  //批次起始索引
      const int channel_idx = blockIdx.y * 8; //通道起始索引

      //每个线程的批次和通道
      const int batchid = batch_idx + tid / 8;
      const int channelid = channel_idx + tid % 8;

      //边界
      if (batchid >= num_tiles || channelid >= ic) return;

      const int collaspsed_idx = batchid * ic + channelid;

      __shared__ float s_tile[256][6];

      float z0,z1,z2,z3,z4,z5,z6;

      //行变换
      for (int w = 0; w < ti.tile_in_w; ++w) {
        // 加载输入数据 - CHWN布局下的内存访问是合并的
        z6 = packed_image[(0 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx];
        
        z0 = 4.0f * z6;
        
        z6 = packed_image[(1 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx];
        
        z1 = -4.0f * z6;
        z2 = 4.0f * z6;
        z3 = -2.0f * z6;
        z4 = 2.0f * z6;
        z5 = 4.0f * z6;
        
        z6 = packed_image[(2 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx];
        
        z0 += -5.0f * z6;
        z1 += -4.0f * z6;
        z2 += -4.0f * z6;
        z3 += -z6;
        z4 += -z6;
        
        z6 = packed_image[(3 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx];
        
        z1 += z6;
        z2 += -z6;
        z3 += 2.0f * z6;
        z4 += -2.0f * z6;
        z5 += -5.0f * z6;
        
        z6 = packed_image[(4 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx];
        
        z0 += z6;
        z1 += z6;
        z2 += z6;
        z3 += z6;
        z4 += z6;
        
        z6 = packed_image[(5 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx];
        
        z5 += z6;
        
        // 存储行变换的中间结果到共享内存
        // 这样做可以减少全局内存访问，提高列变换的效率
        s_tile[tid][0] = z0;
        s_tile[tid][1] = z1;
        s_tile[tid][2] = z2;
        s_tile[tid][3] = z3;
        s_tile[tid][4] = z4;
        s_tile[tid][5] = z5;
        
        // 同步线程，确保所有行变换结果都已写入共享内存
        __syncthreads();
        
        // 写入全局内存
        V[(0 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx] = s_tile[tid][0];
        V[(1 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx] = s_tile[tid][1];
        V[(2 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx] = s_tile[tid][2];
        V[(3 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx] = s_tile[tid][3];
        V[(4 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx] = s_tile[tid][4];
        V[(5 * ti.tile_in_w + w) * num_tiles * ic + collapsed_idx] = s_tile[tid][5];
        
        // 再次同步，确保全局内存写入完成
        __syncthreads();
    }

    //列变换
    for (int h = 0; h < ti.tile_in_h; ++h) {
      // 从全局内存读取行变换的结果
      z6 = V[(h * ti.tile_in_w + 0) * num_tiles * ic + collapsed_idx];
      
      z0 = 4.0f * z6;
      
      z6 = V[(h * ti.tile_in_w + 1) * num_tiles * ic + collapsed_idx];
      
      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;
      
      z6 = V[(h * ti.tile_in_w + 2) * num_tiles * ic + collapsed_idx];
      
      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;
      
      z6 = V[(h * ti.tile_in_w + 3) * num_tiles * ic + collapsed_idx];
      
      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;
      
      z6 = V[(h * ti.tile_in_w + 4) * num_tiles * ic + collapsed_idx];
      
      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;
      
      z6 = V[(h * ti.tile_in_w + 5) * num_tiles * ic + collapsed_idx];
      
      z5 += z6;
      
      // 存储最终的列变换结果到共享内存
      s_tile[tid][0] = z0;
      s_tile[tid][1] = z1;
      s_tile[tid][2] = z2;
      s_tile[tid][3] = z3;
      s_tile[tid][4] = z4;
      s_tile[tid][5] = z5;
      
      // 同步线程
      __syncthreads();
      
      // 写入最终结果到全局内存
      V[(h * 6 + 0) * num_tiles * ic + collapsed_idx] = s_tile[tid][0];
      V[(h * 6 + 1) * num_tiles * ic + collapsed_idx] = s_tile[tid][1];
      V[(h * 6 + 2) * num_tiles * ic + collapsed_idx] = s_tile[tid][2];
      V[(h * 6 + 3) * num_tiles * ic + collapsed_idx] = s_tile[tid][3];
      V[(h * 6 + 4) * num_tiles * ic + collapsed_idx] = s_tile[tid][4];
      V[(h * 6 + 5) * num_tiles * ic + collapsed_idx] = s_tile[tid][5];
      
      // 最后同步
      __syncthreads();
  }

}

void image_packing_cuda(
  float *__restrict__ d_image,
  float *__restrict__ d_packed_image,
  const image_shape_t is,
  const tiling_info_t ti)
{
  // 配置内核执行参数
  // 每个线程处理一个(tile, ic)对
  dim3 blockDim(32, 8);  // 每个线程块处理32*8=256个元素
  
  // 计算网格大小
  // 确保足够的线程覆盖所有的图块和输入通道
  dim3 gridDim(
      (ti.num_tiles + blockDim.x - 1) / blockDim.x,
      (is.ic + blockDim.y - 1) / blockDim.y
  );
  
  // 启动内核
  image_packing_kernel<<<gridDim, blockDim, 0, stream>>>(
      d_image,
      d_packed_image,
      is,
      ti
  );
  
  // 检查内核执行错误
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void image_transform_cuda(
  float *__restrict__ d_packed_image,
  float *__restrict__ d_V,
  const V_shape_t vs,
  const tiling_info_t ti,
  const int64_t collapsed_dim_size)
{
  // 设置核函数执行配置
  const int threadsPerBlock = 256;
  const int blocksNeeded = (collapsed_dim_size + threadsPerBlock - 1) / threadsPerBlock;
  
  // 调用CUDA核函数
  image_transform_kernel<<<blocksNeeded, threadsPerBlock>>>(
      d_packed_image, d_V, vs, ti, collapsed_dim_size);
  
  // 检查错误
  CHECK_CUDA_ERROR(cudaGetLastError());
}

//----------------------------------imgae processing------------------------------------------//


//------------------------------------filter processing---------------------------------------//

__global__ void filter_packing_kernel(
  float *__restrict__ filter,
  float *__restrict__ packed_filter,
  const filter_shape_t fs)
{
  // 计算线程索引 - 基于3D线程块
  int h = blockIdx.x * blockDim.x + threadIdx.x;  // 高度索引
  int w = blockIdx.y * blockDim.y + threadIdx.y;  // 宽度索引
  int oc = blockIdx.z * blockDim.z + threadIdx.z; // 输出通道索引
  
  // 边界检查
  if (h >= fs.h || w >= fs.w || oc >= fs.oc)
      return;
  
  // 计算目标索引前缀
  const int64_t packed_filter_offset = (h * fs.w + w) * fs.oc * fs.ic + oc * fs.ic;
  const int64_t filter_oc_offset = oc * fs.ic * fs.h * fs.w;
  
  // 对每个输入通道执行打包操作
  for (int ic = 0; ic < fs.ic; ++ic) {
      int64_t filter_idx = filter_oc_offset + ic * fs.h * fs.w + h * fs.w + w;
      packed_filter[packed_filter_offset + ic] = filter[filter_idx];
  }
}

void filter_packing_cuda(
  float *__restrict__ d_filter,
  float *__restrict__ d_packed_filter,
  const filter_shape_t fs)
{
  // 定义线程块维度
  dim3 blockDim(2, 2, 32);  // 每个线程块128个线程
  
  // 计算网格维度
  dim3 gridDim(
      (fs.h + blockDim.x - 1) / blockDim.x,
      (fs.w + blockDim.y - 1) / blockDim.y,
      (fs.oc + blockDim.z - 1) / blockDim.z
  );
  
  // 启动核函数
  filter_packing_kernel<<<gridDim, blockDim>>>(d_filter, d_packed_filter, fs);
  
  // 检查错误
  CHECK_CUDA_ERROR(cudaGetLastError());
}

__global__ void filter_transform_kernel(
  float *__restrict__ packed_filter,
  float *__restrict__ U,
  const filter_shape_t fs,
  const int num_oc,      // 输出通道数
  const int num_ic)      // 输入通道数
{
  // 每个线程块处理64*8=512个滤波器块
  const int tid = threadIdx.x;             // 线程ID (0-255)
  const int k_idx = blockIdx.x * 64;       // 输出通道起始索引
  const int c_idx = blockIdx.y * 8;        // 输入通道起始索引
  
  // 计算每个线程负责的输出通道和输入通道
  const int k_offset = tid / 4;            // 每4个线程处理同一输出通道
  const int c_offset = tid % 4;            // 每个线程处理不同输入通道
  
  // 第一个滤波器块的通道索引
  const int k1 = k_idx + k_offset;
  const int c1 = c_idx + c_offset;
  
  // 第二个滤波器块的通道索引 (每个线程处理2个块)
  const int k2 = k_idx + k_offset + 32;    // 第二组输出通道偏移32
  const int c2 = c_idx + c_offset + 4;     // 第二组输入通道偏移4
  
  // 边界检查
  const bool valid1 = (k1 < num_oc && c1 < num_ic);
  const bool valid2 = (k2 < num_oc && c2 < num_ic);
  
  // 计算线性索引
  const int idx1 = k1 * num_ic + c1;
  const int idx2 = k2 * num_ic + c2;
  
  // 共享内存 - 用于存储变换结果
  // 每个线程块有256个线程，每个线程处理2个滤波器块，每个块有6个输出元素
  __shared__ float s_U[512][6];
  
  // 局部变量
  float z0, z1, z2, z3, z4, z5, z6;
  
  if (valid1) {
      // 行变换
      for (int w = 0; w < fs.w; ++w) {
          // 加载滤波器数据
          z6 = packed_filter[(0 * fs.w + w) * num_oc * num_ic + idx1];
          
          // 行变换计算
          z0 = (1.0f / 4.0f) * z6;
          z1 = (-1.0f / 6.0f) * z6;
          z2 = (-1.0f / 6.0f) * z6;
          z3 = (1.0f / 24.0f) * z6;
          z4 = (1.0f / 24.0f) * z6;
          
          z6 = packed_filter[(1 * fs.w + w) * num_oc * num_ic + idx1];
          
          z1 += (-1.0f / 6.0f) * z6;
          z2 += (1.0f / 6.0f) * z6;
          z3 += (1.0f / 12.0f) * z6;
          z4 += (-1.0f / 12.0f) * z6;
          
          z6 = packed_filter[(2 * fs.w + w) * num_oc * num_ic + idx1];
          
          z1 += (-1.0f / 6.0f) * z6;
          z2 += (-1.0f / 6.0f) * z6;
          z3 += (1.0f / 6.0f) * z6;
          z4 += (1.0f / 6.0f) * z6;
          z5 = z6;
          
          // 存储到共享内存
          int s_idx = tid * 2;
          s_U[s_idx][0] = z0;
          s_U[s_idx][1] = z1;
          s_U[s_idx][2] = z2;
          s_U[s_idx][3] = z3;
          s_U[s_idx][4] = z4;
          s_U[s_idx][5] = z5;
          
          // 写入中间结果到全局内存
          U[(0 * fs.w + w) * num_oc * num_ic + idx1] = z0;
          U[(1 * fs.w + w) * num_oc * num_ic + idx1] = z1;
          U[(2 * fs.w + w) * num_oc * num_ic + idx1] = z2;
          U[(3 * fs.w + w) * num_oc * num_ic + idx1] = z3;
          U[(4 * fs.w + w) * num_oc * num_ic + idx1] = z4;
          U[(5 * fs.w + w) * num_oc * num_ic + idx1] = z5;
      }
      
      // 同步线程
      __syncthreads();
      
      // 列变换
      for (int h = 0; h < fs.h; ++h) {
          // 读取行变换结果
          z6 = U[(h * fs.w + 0) * num_oc * num_ic + idx1];
          
          // 列变换计算
          z0 = (1.0f / 4.0f) * z6;
          z1 = (-1.0f / 6.0f) * z6;
          z2 = (-1.0f / 6.0f) * z6;
          z3 = (1.0f / 24.0f) * z6;
          z4 = (1.0f / 24.0f) * z6;
          
          z6 = U[(h * fs.w + 1) * num_oc * num_ic + idx1];
          
          z1 += (-1.0f / 6.0f) * z6;
          z2 += (1.0f / 6.0f) * z6;
          z3 += (1.0f / 12.0f) * z6;
          z4 += (-1.0f / 12.0f) * z6;
          
          z6 = U[(h * fs.w + 2) * num_oc * num_ic + idx1];
          
          z1 += (-1.0f / 6.0f) * z6;
          z2 += (-1.0f / 6.0f) * z6;
          z3 += (1.0f / 6.0f) * z6;
          z4 += (1.0f / 6.0f) * z6;
          z5 = z6;
          
          // 存储最终结果到共享内存
          int s_idx = tid * 2;
          s_U[s_idx][0] = z0;
          s_U[s_idx][1] = z1;
          s_U[s_idx][2] = z2;
          s_U[s_idx][3] = z3;
          s_U[s_idx][4] = z4;
          s_U[s_idx][5] = z5;
          
          // 写入最终结果到全局内存 (CR'S'K布局，保证合并访问)
          U[(h * 6 + 0) * num_oc * num_ic + idx1] = z0;
          U[(h * 6 + 1) * num_oc * num_ic + idx1] = z1;
          U[(h * 6 + 2) * num_oc * num_ic + idx1] = z2;
          U[(h * 6 + 3) * num_oc * num_ic + idx1] = z3;
          U[(h * 6 + 4) * num_oc * num_ic + idx1] = z4;
          U[(h * 6 + 5) * num_oc * num_ic + idx1] = z5;
      }
  }

  if (valid2) {
      // 重复上面的处理，针对第二个滤波器块
      // 行变换
      for (int w = 0; w < fs.w; ++w) {
          // 加载滤波器数据
          z6 = packed_filter[(0 * fs.w + w) * num_oc * num_ic + idx2];
          
          // 行变换计算
          z0 = (1.0f / 4.0f) * z6;
          z1 = (-1.0f / 6.0f) * z6;
          z2 = (-1.0f / 6.0f) * z6;
          z3 = (1.0f / 24.0f) * z6;
          z4 = (1.0f / 24.0f) * z6;
          
          z6 = packed_filter[(1 * fs.w + w) * num_oc * num_ic + idx2];
          
          z1 += (-1.0f / 6.0f) * z6;
          z2 += (1.0f / 6.0f) * z6;
          z3 += (1.0f / 12.0f) * z6;
          z4 += (-1.0f / 12.0f) * z6;
          
          z6 = packed_filter[(2 * fs.w + w) * num_oc * num_ic + idx2];
          
          z1 += (-1.0f / 6.0f) * z6;
          z2 += (-1.0f / 6.0f) * z6;
          z3 += (1.0f / 6.0f) * z6;
          z4 += (1.0f / 6.0f) * z6;
          z5 = z6;
          
          // 存储到共享内存
          int s_idx = tid * 2 + 1;  // 偏移1存储第二个块
          s_U[s_idx][0] = z0;
          s_U[s_idx][1] = z1;
          s_U[s_idx][2] = z2;
          s_U[s_idx][3] = z3;
          s_U[s_idx][4] = z4;
          s_U[s_idx][5] = z5;
          
          // 写入中间结果到全局内存
          U[(0 * fs.w + w) * num_oc * num_ic + idx2] = z0;
          U[(1 * fs.w + w) * num_oc * num_ic + idx2] = z1;
          U[(2 * fs.w + w) * num_oc * num_ic + idx2] = z2;
          U[(3 * fs.w + w) * num_oc * num_ic + idx2] = z3;
          U[(4 * fs.w + w) * num_oc * num_ic + idx2] = z4;
          U[(5 * fs.w + w) * num_oc * num_ic + idx2] = z5;
      }
      
      // 同步线程
      __syncthreads();
      
      // 列变换
      for (int h = 0; h < fs.h; ++h) {
          // 读取行变换结果
          z6 = U[(h * fs.w + 0) * num_oc * num_ic + idx2];
          
          // 列变换计算
          z0 = (1.0f / 4.0f) * z6;
          z1 = (-1.0f / 6.0f) * z6;
          z2 = (-1.0f / 6.0f) * z6;
          z3 = (1.0f / 24.0f) * z6;
          z4 = (1.0f / 24.0f) * z6;
          
          z6 = U[(h * fs.w + 1) * num_oc * num_ic + idx2];
          
          z1 += (-1.0f / 6.0f) * z6;
          z2 += (1.0f / 6.0f) * z6;
          z3 += (1.0f / 12.0f) * z6;
          z4 += (-1.0f / 12.0f) * z6;
          
          z6 = U[(h * fs.w + 2) * num_oc * num_ic + idx2];
          
          z1 += (-1.0f / 6.0f) * z6;
          z2 += (-1.0f / 6.0f) * z6;
          z3 += (1.0f / 6.0f) * z6;
          z4 += (1.0f / 6.0f) * z6;
          z5 = z6;
          
          // 存储最终结果到共享内存
          int s_idx = tid * 2 + 1;
          s_U[s_idx][0] = z0;
          s_U[s_idx][1] = z1;
          s_U[s_idx][2] = z2;
          s_U[s_idx][3] = z3;
          s_U[s_idx][4] = z4;
          s_U[s_idx][5] = z5;
          
          // 写入最终结果到全局内存 (CR'S'K布局)
          U[(h * 6 + 0) * num_oc * num_ic + idx2] = z0;
          U[(h * 6 + 1) * num_oc * num_ic + idx2] = z1;
          U[(h * 6 + 2) * num_oc * num_ic + idx2] = z2;
          U[(h * 6 + 3) * num_oc * num_ic + idx2] = z3;
          U[(h * 6 + 4) * num_oc * num_ic + idx2] = z4;
          U[(h * 6 + 5) * num_oc * num_ic + idx2] = z5;
      }
  }
}

void filter_transform_cuda(
  float *__restrict__ d_packed_filter,
  float *__restrict__ d_U,
  const filter_shape_t fs,
  const int64_t num_oc,
  const int64_t num_ic)
{
  // 线程块大小：256个线程
  const int threadsPerBlock = 256;
  
  // 计算网格大小
  // 每个线程块处理64个输出通道和8个输入通道
  const int k_blocks = (num_oc + 63) / 64;
  const int c_blocks = (num_ic + 7) / 8;
  
  // 创建二维网格
  dim3 gridSize(k_blocks, c_blocks);
  
  // 调用核函数
  filter_transform_kernel<<<gridSize, threadsPerBlock>>>(
      d_packed_filter, d_U, fs, num_oc, num_ic);
  
  // 检查错误
  CHECK_CUDA_ERROR(cudaGetLastError());
}

//----------------------------------filter processing-----------------------------------------//




//----------------------------------output processing-----------------------------------------//

__global__ void output_transform_kernel(
  float *__restrict__ M,
  float *__restrict__ Y,
  const tiling_info_t ti,
  const int num_tiles,
  const int oc)
{
  // 每个线程块处理64*8=512个输出块 (充分利用L40的128KB共享内存)
  const int tid = threadIdx.x;             // 线程ID (0-255)
  const int tile_idx = blockIdx.x * 64;    // 图块起始索引 (增加到64个图块/块)
  const int k_idx = blockIdx.y * 8;        // 输出通道起始索引 (增加到8个通道/块)
  
  // 计算每个线程负责的图块和输出通道
  const int tile_offset = tid / 8;         // 每8个线程处理一个图块
  const int k_offset = tid % 8;            // 每个线程处理一个输出通道
  
  const int tile = tile_idx + tile_offset;
  const int k = k_idx + k_offset;
  
  // 边界检查
  const bool valid = (tile < num_tiles && k < oc);
  
  // 计算线性索引
  const int idx = tile * oc + k;
  
  // 分配共享内存 - 为L40优化 (共享内存大小为128KB)
  // 由于共享内存足够大，可以在一次执行中完成所有变换，不再需要分轮
  __shared__ float s_buffer[512][36]; // 512个图块-通道对，每个6x6 (含填充)
  
  // 局部变量
  float z0, z1, z2, z3, z4, z5;
  
  // 只有在有效范围内的线程才执行
  if (valid) {
      // 行变换 - 每个线程处理特定的行
      for (int h = 0; h < ti.tile_in_h; ++h) {
          for (int w = 0; w < ti.tile_in_w; ++w) {
              // 从全局内存加载数据
              float data = M[(h * ti.tile_in_w + w) * num_tiles * oc + idx];
              
              // 存储到共享内存
              s_buffer[tid][h * 6 + w] = data;
          }
      }
      
      // 同步所有线程
      __syncthreads();
      
      // 行变换计算
      for (int h = 0; h < ti.tile_in_h; ++h) {
          // 从共享内存加载数据
          z0 = s_buffer[tid][h * 6 + 0];
          z1 = s_buffer[tid][h * 6 + 1];
          z2 = s_buffer[tid][h * 6 + 2];
          z3 = s_buffer[tid][h * 6 + 3];
          z4 = s_buffer[tid][h * 6 + 4];
          z5 = s_buffer[tid][h * 6 + 5];
          
          // 行变换计算
          float t0 = z0 + z1 + z2 + z3 + z4;
          float t1 = z1 - z2 + 2.0f * z3 - 2.0f * z4;
          float t2 = z1 + z2 + 4.0f * z3 + 4.0f * z4;
          float t3 = z1 - z2 + 8.0f * z3 - 8.0f * z4 + z5;
          
          // 存回共享内存
          s_buffer[tid][h * 6 + 0] = t0;
          s_buffer[tid][h * 6 + 1] = t1;
          s_buffer[tid][h * 6 + 2] = t2;
          s_buffer[tid][h * 6 + 3] = t3;
      }
      
      // 同步所有线程
      __syncthreads();
      
      // 列变换
      for (int w = 0; w < ti.tile_out_w; ++w) {
          // 从共享内存加载数据
          z0 = s_buffer[tid][0 * 6 + w];
          z1 = s_buffer[tid][1 * 6 + w];
          z2 = s_buffer[tid][2 * 6 + w];
          z3 = s_buffer[tid][3 * 6 + w];
          z4 = s_buffer[tid][4 * 6 + w];
          z5 = s_buffer[tid][5 * 6 + w];
          
          // 列变换计算
          float t0 = z0 + z1 + z2 + z3 + z4;
          float t1 = z1 - z2 + 2.0f * z3 - 2.0f * z4;
          float t2 = z1 + z2 + 4.0f * z3 + 4.0f * z4;
          float t3 = z1 - z2 + 8.0f * z3 - 8.0f * z4 + z5;
          
          // 写入最终结果到全局内存 - 只保留4x4输出区域
          if (w < ti.tile_out_w) {
              Y[(0 * ti.tile_out_w + w) * num_tiles * oc + idx] = t0;
              if (1 < ti.tile_out_h)
                  Y[(1 * ti.tile_out_w + w) * num_tiles * oc + idx] = t1;
              if (2 < ti.tile_out_h)
                  Y[(2 * ti.tile_out_w + w) * num_tiles * oc + idx] = t2;
              if (3 < ti.tile_out_h)
                  Y[(3 * ti.tile_out_w + w) * num_tiles * oc + idx] = t3;
          }
      }
  }
}

__global__ void output_unpacking_kernel(
  float *__restrict__ Y,
  float *__restrict__ out,
  const out_shape_t os,
  const tiling_info_t ti)
{
  // 计算线程索引
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int w = blockIdx.y * blockDim.y + threadIdx.y;
  int oc = blockIdx.z * blockDim.z + threadIdx.z;
  
  // 边界检查
  if (h >= ti.tile_out_h || w >= ti.tile_out_w || oc >= os.oc)
      return;
  
  // 遍历所有图块，将结果写入到输出
  for (int tile = 0; tile < ti.num_tiles; ++tile) {
      // 计算图块在输出中的实际位置
      tile_index_t tidx = get_tile_index(tile, ti);
      int batch = tidx.b;
      int hh = tidx.th;
      int ww = tidx.tw;
      
      // 计算最终输出位置
      int out_h = hh * 4 + h;
      int out_w = ww * 4 + w;
      
      // 检查是否在输出范围内
      if (out_h < os.h && out_w < os.w) {
          // 计算Y中的源索引
          int64_t src_idx = (h * ti.tile_out_w + w) * ti.num_tiles * os.oc + tile * os.oc + oc;
          
          // 计算输出张量中的目标索引
          int64_t dst_idx = ((batch * os.oc + oc) * os.h + out_h) * os.w + out_w;
          
          // 写入结果
          out[dst_idx] = Y[src_idx];
      }
  }
}


void output_transform_cuda(
  float *__restrict__ d_M,
  float *__restrict__ d_Y,
  const tiling_info_t ti,
  const int64_t num_tiles,
  const int64_t oc)
{
  // 设置线程块大小为256
  const int threadsPerBlock = 256;
  
  // 计算网格大小
  // 每个线程块处理64个图块和8个输出通道 (针对L40的优化)
  const int tile_blocks = (num_tiles + 63) / 64;
  const int oc_blocks = (oc + 7) / 8;
  
  // 创建二维网格
  dim3 gridSize(tile_blocks, oc_blocks);
  
  // 调用输出变换核函数
  output_transform_kernel<<<gridSize, threadsPerBlock>>>(
      d_M, d_Y, ti, num_tiles, oc);
  
  // 检查错误
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void output_unpacking_cuda(
  float *__restrict__ d_Y,
  float *__restrict__ d_out,
  const out_shape_t os,
  const tiling_info_t ti)
{
  // 设置线程块维度
  dim3 blockDim(4, 4, 16);  // 每个线程块256个线程
  
  // 计算网格维度
  dim3 gridDim(
      (ti.tile_out_h + blockDim.x - 1) / blockDim.x,
      (ti.tile_out_w + blockDim.y - 1) / blockDim.y,
      (os.oc + blockDim.z - 1) / blockDim.z
  );
  
  // 调用输出解包核函数
  output_unpacking_kernel<<<gridDim, blockDim>>>(
      d_Y, d_out, os, ti);
  
  // 检查错误
  CHECK_CUDA_ERROR(cudaGetLastError());
}

//----------------------------------output processing-----------------------------------------//

//-----------------------------------gemm processing------------------------------------------//





//-----------------------------------gemm processing------------------------------------------//



void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  float z0, z1, z2, z3, z4, z5, z6;

  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z6 = packed_image_tensor[0][w][idx];

      z0 = 4.0f * z6;

      z6 = packed_image_tensor[1][w][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = packed_image_tensor[2][w][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packed_image_tensor[3][w][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packed_image_tensor[4][w][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packed_image_tensor[5][w][idx];

      z5 += z6;

      V_tensor[0][w][idx] = z0;
      V_tensor[1][w][idx] = z1;
      V_tensor[2][w][idx] = z2;
      V_tensor[3][w][idx] = z3;
      V_tensor[4][w][idx] = z4;
      V_tensor[5][w][idx] = z5;
    }

    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
      z6 = V_tensor[h][0][idx];

      z0 = 4.0f * z6;

      z6 = V_tensor[h][1][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = V_tensor[h][2][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = V_tensor[h][3][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = V_tensor[h][4][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = V_tensor[h][5][idx];

      z5 += z6;

      V_tensor[h][0][idx] = z0;
      V_tensor[h][1][idx] = z1;
      V_tensor[h][2][idx] = z2;
      V_tensor[h][3][idx] = z3;
      V_tensor[h][4][idx] = z4;
      V_tensor[h][5][idx] = z5;
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

  float z0, z1, z2, z3, z4, z5, z6;

  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    for (int64_t w = 0; w < fs.w; ++w) {
      z6 = packed_filter_tensor[0][w][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packed_filter_tensor[1][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packed_filter_tensor[2][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[0][w][idx] = z0;
      U_tensor[1][w][idx] = z1;
      U_tensor[2][w][idx] = z2;
      U_tensor[3][w][idx] = z3;
      U_tensor[4][w][idx] = z4;
      U_tensor[5][w][idx] = z5;
    }

    for (int64_t h = 0; h < us.h; ++h) {
      z6 = U_tensor[h][0][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U_tensor[h][1][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U_tensor[h][2][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[h][0][idx] = z0;
      U_tensor[h][1][idx] = z1;
      U_tensor[h][2][idx] = z2;
      U_tensor[h][3][idx] = z3;
      U_tensor[h][4][idx] = z4;
      U_tensor[h][5][idx] = z5;
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
  float z0, z1, z2, z3, z4;

  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z4 = M_tensor[0][w][idx];
      z0 = z4;

      z4 = M_tensor[1][w][idx];
      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = M_tensor[2][w][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = M_tensor[3][w][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = M_tensor[4][w][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = M_tensor[5][w][idx];
      z3 += z4;

      Y_tensor[0][w][idx] = z0;
      Y_tensor[1][w][idx] = z1;
      Y_tensor[2][w][idx] = z2;
      Y_tensor[3][w][idx] = z3;
    }

    for (int64_t h = 0; h < ti.tile_out_h; ++h) {
      z4 = Y_tensor[h][0][idx];

      z0 = z4;

      z4 = Y_tensor[h][1][idx];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = Y_tensor[h][2][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = Y_tensor[h][3][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = Y_tensor[h][4][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = Y_tensor[h][5][idx];

      z3 += z4;

      Y_tensor[h][0][idx] = z0;
      Y_tensor[h][1][idx] = z1;
      Y_tensor[h][2][idx] = z2;
      Y_tensor[h][3][idx] = z3;
    }
  }
}

void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

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

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      C_tensor[n][m] = 0;
      for (int64_t k = 0; k < K; ++k) {
        C_tensor[n][m] += A_tensor[m][k] * B_tensor[n][k];
      }
    }
  }
}

void winograd_convolution(
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

    if(!init) {
        initCUDA();
        cublasStreamCreate(&stream);
        cublasCreate(&handle);
        cublasSetStream(handle, stream);

        init = true;
    }

    const size_t image_size = sizeof(float) * batch_num * input_channel_num * image_height * image_width;
    const size_t filter_size = output_channel_num * input_channel_num * FLT_H * FLT_W;
    const size_t out_size = sizeof(float) * os.bs * os.oc * os.h * os.w;

    const size_t packed_image_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic;
    const size_t packed_filter_size = sizeof(float) * fs.h * fs.w * fs.oc * fs.ic;
    const size_t V_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic;
    const size_t U_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic;
    const size_t M_size = sizeof(float) * ti.tile_in_h * ti.tile_in_w * fs.oc * ti.num_tiles;
    const size_t Y_size = sizeof(float) * ti.tile_out_h * ti.tile_out_w * fs.oc * ti.num_tiles;

    ensure_mem_size((void **)&g_image, &gimage_cur_size,image_size,false);
    ensure_mem_size((void **)&g_filter, &gfilter_cur_size,filter_size,false);
    ensure_mem_size((void **)&g_out, &gout_cur_size,out_size,false);

    ensure_mem_size((void **)&g_packed_image, &g_packed_image_size, packed_image_size, false);
    ensure_mem_size((void **)&g_packed_filter, &g_packed_filter_size, packed_filter_size, false);
    ensure_mem_size((void **)&g_U, &g_U_size, U_size, false);
    ensure_mem_size((void **)&g_V, &g_V_size, V_size, false);
    ensure_mem_size((void **)&g_M, &g_M_size, M_size, false);
    ensure_mem_size((void **)&g_Y, &g_Y_size, Y_size, false);

    ensure_mem_size((void **)&image, &pimage_cur_size,image_size,true);
    ensure_mem_size((void **)&filter, &pfilter_cur_size,filter_size,true);
    ensure_mem_size((void **)&out, &pout_cur_size,out_size,true);

    memcpy(pin_image, image, image_size);
    memcpy(pin_filter, filter, filter_size);

    cudaMemcpyAsync(g_image, pin_image, image_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_filter, pin_filter, filter_size, cudaMemcpyHostToDevice, stream);

    filter_packing_cuda(g_filter, g_packed_filter, fs);
    image_packing_cuda(g_image, g_packed_image, is, ti);

    filter_transform_cuda(g_packed_filter, g_U, fs, fs.oc, fs.ic);
    image_transform_cuda(g_packed_image, g_V, vs, ti, ti.num_tiles * is.ic);

    int batch_count = ti.tile_in_h * ti.tile_in_w;

    int m = vs.num_tiles;
    int n = us.oc;
    int k = us.ic;

    const long long A_size = vs.num_tiles * vs.ic;
    const long long B_size = us.oc * us.ic;
    const long long C_size = vs.num_tiles * us.oc;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasGemmStridedBatchedEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      m,
      n,
      k,
      &alpha,
      g_V,
      CUDA_R_32F,
      k,
      A_size,
      g_U,
      CUDA_R_32F,
      k,
      B_size,
      &beta,
      g_M,
      CUDA_R_32F,
      m,
      C_size,
      batch_count,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT,
    );

    output_transform_cuda(g_M, g_Y, ti, vs.num_tiles, us.oc);
    output_unpacking_cuda(g_Y, g_out, os, ti);

    cudaMemcpyAsync(pin_out, g_out, out_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    memcpy(out, pin_out, out_size);  

}
