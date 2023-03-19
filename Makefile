CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
CCFLAGS     :=
LDFLAGS     :=
GENCODE_FLAGS := -gencode arch=compute_75,code=compute_75 -rdc=true -O2


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

test_c: test_c.o blake3_specific_target_dbg.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	# g++ $+ -I../c -lblake3 -L../c -Wl,-rpath=../c -o test_c

blake3_test: test.o blake3_cuda.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./blake3_test

clean:
	rm -f blake3_cuda blake3_cuda.o blake3_test test.o test_c test

clobber: clean
