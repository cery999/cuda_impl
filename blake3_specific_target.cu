#pragma once
// includes, system
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;

#ifdef __cplusplus
extern "C" {
#endif

#define BLAKE3_VERSION_STRING "8.8.8"
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_MAX_DEPTH 54

#define INPUT_LEN 180
#define 

static void *pined_inp, *pined_target;

__global__ special_launch(uint8_t *header, size_t start, size_t end,
        size_t stride) {

    uint32_t CV[8];
    uint32_t M[16];

}

void special_cuda_target(uint8_t *header, size_t start, size_t end,
                         size_t stride, uint8_t target[32]) {
  checkCudaErrors(cudaProfilerStart());
  cudaMemcpyAsync(pined_inp, header, INPUT_LEN * , cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(pined_target, header, INPUT_LEN, cudaMemcpyHostToDevice, 0);

  checkCudaErrors(cudaProfilerStop());
}

extern "C" void pre_allocate() {
  checkCudaErrors(cudaMalloc((void **)&pined_inp, INPUT_LEN));
}

extern "C" void post_free() { checkCudaErrors(cudaFree(pined_inp)); }

#ifdef __cplusplus
}
#endif
