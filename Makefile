CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
CCFLAGS     :=
LDFLAGS     :=
GENCODE_FLAGS := -gencode arch=compute_75,code=compute_75 -rdc=true -O3


INCLUDES  := -I/home/cery/cuda-samples-master/Common -I../c/
LIBRARIES :=


all: build

build: blake3_test test_c

blake3_cuda.o:blake3_cuda.cu 
	$(EXEC) $(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

blake3_specific_target.o:blake3_specific_target.cu 
	$(EXEC) $(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

blake3_specific_target_dbg.o:blake3_specific_target_test.cu 
	$(EXEC) $(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

blake3_cuda: blake3_cuda.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

test.o: test.cpp 
	$(EXEC) $(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

test_c.o: test_c.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

test_c: test_c.o blake3_specific_target_dbg.o compile_blake3
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ test_c.o blake3_specific_target_dbg.o blake3.o blake3_dispatch.o blake3_portable.o $(LIBRARIES)
	# g++ $+ -I../c -lblake3 -L../c -Wl,-rpath=../c -o test_c
	

EXTRAFLAGS=-Wa,--noexecstack -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512

compile_blake3: ../c/blake3.c ../c/blake3_dispatch.c ../c/blake3_portable.c 
	gcc -c $+ -I../c -O3 -Wall -Wextra -std=c11 -pedantic -fstack-protector-strong -D_FORTIFY_SOURCE=2  -Wl,-z,relro,-z,now  $(EXTRAFLAGS) 

blake3_test: test.o blake3_cuda.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./blake3_test

clean:
	rm -f blake3_cuda blake3_cuda.o blake3_test test.o test_c test

clobber: clean
