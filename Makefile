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

build: blake3-cuda

blake3-cuda.o:blake3-cuda.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<


blake3-cuda: blake3-cuda.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	
run: build
	$(EXEC) ./blake3-cuda

testrun: build
	$(EXEC) ./blake3-cuda

clean:
	rm -f blake3-cuda blake3-cuda.o

clobber: clean
