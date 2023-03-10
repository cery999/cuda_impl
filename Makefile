CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
CCFLAGS     :=
LDFLAGS     :=
GENCODE_FLAGS := -gencode arch=compute_75,code=compute_75


INCLUDES  := -I/home/cery/cuda-samples-master/Common
LIBRARIES :=


all: build

build: blake3_cuda

blake3_cuda.o:blake3_cuda.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<


blake3_cuda: blake3_cuda.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	
run: build
	$(EXEC) ./blake3_cuda

testrun: build
	$(EXEC) ./blake3_cuda

clean:
	rm -f blake3_cuda blake3_cuda.o

clobber: clean
