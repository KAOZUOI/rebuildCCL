CUDA_HOME ?= /root/kernel_dev/curr/ucxx-dev/.pixi/envs/default
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_INCLUDE = -I$(CUDA_HOME)/include
CUDA_LIB = -L$(CUDA_HOME)/lib64 -lcudart

# Compiler flags
NVCC_FLAGS = -O3 -std=c++14 $(CUDA_INCLUDE)

# Target executables
TARGETS = kernel_alltoall kernel_alltoall_optimized

all: $(TARGETS)

kernel_alltoall: kernel_alltoall.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIB)

kernel_alltoall_optimized: kernel_alltoall_optimized.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_LIB)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
