# 修改后的兼容性Makefile
CFLAG = -O3 -g -Wall -mavx -fopenmp -mfma -mavx2 -mavx512f -mavx512dq -mavx512vl -mavx512bw
CUDA_INCLUDES = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcublas
NVCC_FLAGS = -O3 -std=c++11

all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} ${CUDA_INCLUDES} ${CUDA_LIBS} -o winograd

# 添加一个专门针对节点间兼容性的目标

compat:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} -I/usr/local/cuda/include \
	-Wl,--allow-shlib-undefined \
	-Wl,--unresolved-symbols=ignore-all \
	-o winograd

clean:
	rm -f winograd winograd