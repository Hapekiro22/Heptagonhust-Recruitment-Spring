# 使用nvcc编译器
# 主机编译器选项（需要通过-Xcompiler传递）
HOST_FLAGS = -O3 -g -mavx -fopenmp -mfma -mavx2

# CUDA路径和库
CUDA_PATH = /usr/local/cuda
CUDA_INCLUDES = -I${CUDA_PATH}/include
CUDA_LIBS = -L${CUDA_PATH}/targets/x86_64-linux/lib -lcudart -lcublas -lcudnn

# NVCC原生选项
NVCC_FLAGS = -O3 -std=c++11 -gencode arch=compute_89,code=sm_89 -Xcompiler "${HOST_FLAGS}"

all:
	nvcc winograd.cc driver.cc -x cu ${NVCC_FLAGS} ${CUDA_INCLUDES} ${CUDA_LIBS} -o winograd

debug:
	nvcc winograd.cc driver.cc -x cu -G -g -std=c++11 -gencode arch=compute_89,code=sm_89 -Xcompiler "-O0 -g -mavx -fopenmp -mfma -mavx2" ${CUDA_INCLUDES} ${CUDA_LIBS} -o winograd_debug

clean:
	rm -f winograd winograd_debug