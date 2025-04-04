#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <string.h>

// 辅助函数声明
void printDeviceInfo();
void testSimpleConversion();
bool testGemmWithParams(int m, int n, int k, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
void printMatrix(__nv_bfloat16* matrix, int rows, int cols, const char* name);
void printMatrixFloat(float* matrix, int rows, int cols, const char* name);
void generateRandomMatrix(float* matrix, int size);

int main(int argc, char** argv) {
    // 打印设备信息
    printDeviceInfo();
    
    // 测试BF16简单转换
    testSimpleConversion();
    
    // GEMM参数
    int m_values[] = {2, 4, 16, 64, 128};
    int n_values[] = {2, 4, 16, 64, 128};
    int k_values[] = {2, 4, 16, 64, 128};
    
    // 计算类型
    cublasComputeType_t computeTypes[] = {
        CUBLAS_COMPUTE_32F, 
        CUBLAS_COMPUTE_32F_FAST_16BF
    };
    const char* computeTypeNames[] = {
        "CUBLAS_COMPUTE_32F", 
        "CUBLAS_COMPUTE_32F_FAST_16BF"
    };
    
    // 算法选择
    cublasGemmAlgo_t algos[] = {
        CUBLAS_GEMM_DEFAULT,
        CUBLAS_GEMM_ALGO0,
        CUBLAS_GEMM_ALGO1,
        CUBLAS_GEMM_ALGO2,
        CUBLAS_GEMM_ALGO3
    };
    const char* algoNames[] = {
        "CUBLAS_GEMM_DEFAULT",
        "CUBLAS_GEMM_ALGO0",
        "CUBLAS_GEMM_ALGO1",
        "CUBLAS_GEMM_ALGO2",
        "CUBLAS_GEMM_ALGO3"
    };
    
    // 如果命令行指定了具体参数，则只测试这组参数
    if (argc >= 6) {
        int m = atoi(argv[1]);
        int n = atoi(argv[2]);
        int k = atoi(argv[3]);
        int computeType = atoi(argv[4]);
        int algo = atoi(argv[5]);
        
        printf("使用命令行参数: m=%d, n=%d, k=%d, computeType=%d, algo=%d\n", 
               m, n, k, computeType, algo);
        
        if (computeType >= 0 && computeType < 2 && algo >= 0 && algo < 5) {
            testGemmWithParams(m, n, k, computeTypes[computeType], algos[algo]);
        } else {
            printf("无效的计算类型或算法索引\n");
        }
        return 0;
    }
    
    // 测试所有参数组合
    printf("\n=========== 开始全面 BF16 GEMM 测试 ===========\n");
    printf("将测试 %lu x %lu x %lu x %lu x %lu = %lu 种参数组合\n",
           sizeof(m_values)/sizeof(int), sizeof(n_values)/sizeof(int), sizeof(k_values)/sizeof(int),
           sizeof(computeTypes)/sizeof(cublasComputeType_t), sizeof(algos)/sizeof(cublasGemmAlgo_t),
           sizeof(m_values)/sizeof(int) * sizeof(n_values)/sizeof(int) * sizeof(k_values)/sizeof(int) *
           sizeof(computeTypes)/sizeof(cublasComputeType_t) * sizeof(algos)/sizeof(cublasGemmAlgo_t));
           
    // 记录通过和失败的测试
    int passed = 0;
    int failed = 0;
    
    // 保存通过的组合
    struct SuccessfulConfig {
        int m, n, k;
        int computeTypeIdx;
        int algoIdx;
    };
    
    SuccessfulConfig successConfigs[100];  // 假设不超过100个成功案例
    int successCount = 0;
    
    // 测试所有组合
    for (int midx = 0; midx < sizeof(m_values)/sizeof(int); midx++) {
        for (int nidx = 0; nidx < sizeof(n_values)/sizeof(int); nidx++) {
            for (int kidx = 0; kidx < sizeof(k_values)/sizeof(int); kidx++) {
                for (int ctidx = 0; ctidx < sizeof(computeTypes)/sizeof(cublasComputeType_t); ctidx++) {
                    for (int aidx = 0; aidx < sizeof(algos)/sizeof(cublasGemmAlgo_t); aidx++) {
                        int m = m_values[midx];
                        int n = n_values[nidx];
                        int k = k_values[kidx];
                        cublasComputeType_t computeType = computeTypes[ctidx];
                        cublasGemmAlgo_t algo = algos[aidx];
                        
                        printf("\n测试: m=%d, n=%d, k=%d, computeType=%s, algo=%s\n", 
                               m, n, k, computeTypeNames[ctidx], algoNames[aidx]);
                        
                        bool success = testGemmWithParams(m, n, k, computeType, algo);
                        if (success) {
                            printf("✅ 测试通过!\n");
                            passed++;
                            
                            // 保存成功配置
                            if (successCount < 100) {
                                successConfigs[successCount].m = m;
                                successConfigs[successCount].n = n;
                                successConfigs[successCount].k = k;
                                successConfigs[successCount].computeTypeIdx = ctidx;
                                successConfigs[successCount].algoIdx = aidx;
                                successCount++;
                            }
                        } else {
                            printf("❌ 测试失败...\n");
                            failed++;
                        }
                    }
                }
            }
        }
    }
    
    printf("\n=========== 测试结果汇总 ===========\n");
    printf("通过: %d, 失败: %d, 总计: %d\n", passed, failed, passed + failed);
    
    if (passed > 0) {
        printf("\n成功的参数组合:\n");
        for (int i = 0; i < successCount; i++) {
            printf("配置 #%d: m=%d, n=%d, k=%d, computeType=%s, algo=%s\n", 
                   i+1, 
                   successConfigs[i].m, 
                   successConfigs[i].n, 
                   successConfigs[i].k, 
                   computeTypeNames[successConfigs[i].computeTypeIdx],
                   algoNames[successConfigs[i].algoIdx]);
        }
    }
    
    return 0;
}

// 打印设备信息
void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("检测到 %d 个CUDA设备:\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("设备 #%d: %s\n", i, prop.name);
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  多处理器数量: %d\n", prop.multiProcessorCount);
        printf("  最大线程数/块: %d\n", prop.maxThreadsPerBlock);
        printf("  最大共享内存/块: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  时钟频率: %d MHz\n", prop.clockRate / 1000);
    }
    
    printf("\n");
}

// 测试简单的BF16转换
void testSimpleConversion() {
    printf("测试BF16基本转换...\n");
    
    float values[] = {-210.0f, -40.0f, 0.0f, 0.062f, 1.0f, 40.0f, 210.0f};
    
    for (int i = 0; i < sizeof(values) / sizeof(float); i++) {
        __nv_bfloat16 bf16 = __float2bfloat16(values[i]);
        float back = __bfloat162float(bf16);
        
        printf("原始值: %f, BF16转换后: %f, 差异: %f\n", 
               values[i], back, fabsf(values[i] - back));
    }
    
    printf("\n");
}

// 使用指定参数测试GEMM
bool testGemmWithParams(int m, int n, int k, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    // 创建矩阵
    float* A = (float*)malloc(m * k * sizeof(float));
    float* B = (float*)malloc(k * n * sizeof(float));
    float* C = (float*)malloc(m * n * sizeof(float));
    float* C_expected = (float*)malloc(m * n * sizeof(float));
    
    // 填充随机数据
    generateRandomMatrix(A, m * k);
    generateRandomMatrix(B, k * n);
    memset(C, 0, m * n * sizeof(float));
    
    // 计算预期结果 (CPU)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C_expected[i * n + j] = sum;
        }
    }
    
    // 创建BF16矩阵
    __nv_bfloat16* A_bf16 = (__nv_bfloat16*)malloc(m * k * sizeof(__nv_bfloat16));
    __nv_bfloat16* B_bf16 = (__nv_bfloat16*)malloc(k * n * sizeof(__nv_bfloat16));
    __nv_bfloat16* C_bf16 = (__nv_bfloat16*)malloc(m * n * sizeof(__nv_bfloat16));
    
    // 转换为BF16
    for (int i = 0; i < m * k; i++) {
        A_bf16[i] = __float2bfloat16(A[i]);
    }
    for (int i = 0; i < k * n; i++) {
        B_bf16[i] = __float2bfloat16(B[i]);
    }
    for (int i = 0; i < m * n; i++) {
        C_bf16[i] = __float2bfloat16(0.0f);
    }
    
    // 分配CUDA内存
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, k * n * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, m * n * sizeof(__nv_bfloat16));
    
    // 复制数据到设备
    cudaMemcpy(d_A, A_bf16, m * k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_bf16, k * n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_bf16, m * n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // 执行GEMM
    __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    __nv_bfloat16 beta = __float2bfloat16(0.0f);
    
    cudaError_t cudaStatus;
    cublasStatus_t status;
    
    // 使用cuBLAS调用GEMM
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_B, CUDA_R_16BF, n,
        d_A, CUDA_R_16BF, k,
        &beta,
        d_C, CUDA_R_16BF, n,
        computeType,
        algo
    );
    
    cudaStatus = cudaDeviceSynchronize();
    
    // 检查错误
    if (status != CUBLAS_STATUS_SUCCESS || cudaStatus != cudaSuccess) {
        printf("cuBLAS错误: %d, CUDA错误: %s\n", status, cudaGetErrorString(cudaStatus));
        return false;
    }
    
    // 复制结果回主机
    cudaMemcpy(C_bf16, d_C, m * n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // 转换回float
    for (int i = 0; i < m * n; i++) {
        C[i] = __bfloat162float(C_bf16[i]);
    }
    
    // 打印小尺寸矩阵结果
    if (m <= 4 && n <= 4) {
        printMatrixFloat(A, m, k, "A");
        printMatrixFloat(B, k, n, "B");
        printMatrixFloat(C, m, n, "C (BF16 GEMM结果)");
        printMatrixFloat(C_expected, m, n, "C_expected (预期结果)");
    }
    
    // 验证结果
    bool success = true;
    float max_diff = 0.0f;
    float sum = 0.0f;
    
    for (int i = 0; i < m * n; i++) {
        float diff = fabsf(C[i] - C_expected[i]);
        max_diff = fmaxf(max_diff, diff);
        sum += fabsf(C[i]);
        
        // 检测零结果 (所有结果都是零会被认为失败)
        if (i == 0 && C[i] == 0.0f) {
            bool all_zeros = true;
            for (int j = 1; j < m * n; j++) {
                if (C[j] != 0.0f) {
                    all_zeros = false;
                    break;
                }
            }
            
            if (all_zeros) {
                printf("错误: 返回全零矩阵！\n");
                success = false;
            }
        }
        
        // 允许一定的误差范围 (BF16精度有限)
        float rel_error = (C_expected[i] != 0.0f) ? diff / fabsf(C_expected[i]) : diff;
        if (rel_error > 0.05f && diff > 0.1f) {  // 5%相对误差或0.1绝对误差
            printf("误差过大: C[%d] = %f, 期望 %f, 差异 %f, 相对误差 %f\n", 
                   i, C[i], C_expected[i], diff, rel_error);
            success = false;
        }
    }
    
    printf("最大误差: %f, 输出总和: %f\n", max_diff, sum);
    
    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    
    free(A);
    free(B);
    free(C);
    free(C_expected);
    free(A_bf16);
    free(B_bf16);
    free(C_bf16);
    
    return success && sum > 0.0f;  // 确保输出不全为零
}

// 打印矩阵 (BF16格式)
void printMatrix(__nv_bfloat16* matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", __bfloat162float(matrix[i * cols + j]));
        }
        printf("\n");
    }
    printf("\n");
}

// 打印矩阵 (float格式)
void printMatrixFloat(float* matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 生成随机矩阵
void generateRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // -1到1之间的随机数
    }
}