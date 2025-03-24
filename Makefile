CFLAG = -O3 -g -Wall -mavx -fopenmp -mfma -mavx2

all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} -o winograd

clean:
	rm -f winograd