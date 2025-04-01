# 使用nvcc编译器
CFLAG = -O3 -g -Wall -mavx -fopenmp -mfma -mavx2
CUDA_PATH = /usr/local/cuda
CUDA_INCLUDES = -I${CUDA_PATH}/include
CUDA_LIBS = -L${CUDA_PATH}/targets/x86_64-linux/lib -lcudart -lcublas -lcudnn
NVCC_FLAGS = -O3 -std=c++11 -gencode arch=compute_89,code=sm_89

all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} ${CUDA_INCLUDES} ${CUDA_LIBS} -o winograd

clean:
	rm -f winograd