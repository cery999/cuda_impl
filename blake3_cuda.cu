// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h> // helper functions for CUDA error checking and initialization
#include <helper_functions.h> // helper utility functions

#ifdef __cplusplus
extern "C" {
#endif

#define BLAKE3_VERSION_STRING "1.3.3"
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_MAX_DEPTH 54

__constant__ uint32_t IV[8]{0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL,
                            0xA54FF53AUL, 0x510E527FUL, 0x9B05688CUL,
                            0x1F83D9ABUL, 0x5BE0CD19UL};
__constant__ uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};
__device__ uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
}

inline __device__ __host__ void store32(void *dst, uint32_t w) {
  uint8_t *p = (uint8_t *)dst;
  p[0] = (uint8_t)(w >> 0);
  p[1] = (uint8_t)(w >> 8);
  p[2] = (uint8_t)(w >> 16);
  p[3] = (uint8_t)(w >> 24);
}

inline void __host__ __device__ store_cv_words(uint8_t bytes_out[32],
                                               uint32_t cv_words[8]) {
  store32(&bytes_out[0 * 4], cv_words[0]);
  store32(&bytes_out[1 * 4], cv_words[1]);
  store32(&bytes_out[2 * 4], cv_words[2]);
  store32(&bytes_out[3 * 4], cv_words[3]);
  store32(&bytes_out[4 * 4], cv_words[4]);
  store32(&bytes_out[5 * 4], cv_words[5]);
  store32(&bytes_out[6 * 4], cv_words[6]);
  store32(&bytes_out[7 * 4], cv_words[7]);
}

__device__ void g(uint32_t *state, size_t a, size_t b, size_t c, size_t d,
                  uint32_t x, uint32_t y) {
  state[a] = state[a] + state[b] + x;
  /* state[d] = rotr32(state[d] ^ state[a], 16); */
  // use __byte_perm for 8 bytes align
  uint32_t t = state[d] ^ state[a];
  state[d] = __byte_perm(t, t, 2 | (3 << 4) | 0 | (1 << 12));
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 12);
  state[a] = state[a] + state[b] + y;
  /* state[d] = rotr32(state[d] ^ state[a], 8); */
  // use __byte_perm for 8 bytes align
  t = state[d] ^ state[a];
  state[d] = __byte_perm(t, t, 3 | 0 | (1 << 8) | (1 << 12));
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 7);
}
__device__ void round_fn(uint32_t *state, const uint32_t *msg, size_t round) {
  // Select the message schedule based on the round.
  const uint8_t *schedule = MSG_SCHEDULE[round];

  // Mix the columns.
  g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
  g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
  g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
  g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

  // Mix the rows.
  g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
  g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
  g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
  g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

__global__ void blake3_compress_in_place_cuda_kernel(uint32_t *cv,
                                                     const uint8_t *block,
                                                     uint8_t block_len,
                                                     uint64_t counter,
                                                     uint8_t flags) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t state[16];       // v0 - v15
  uint32_t block_words[16]; // m0 - m15
#pragma unroll
  for (auto i = 0; i < 16; i++) {
    block_words[i] = *((uint32_t *)(&block + i * 4));
  }

#pragma unroll
  for (auto i = 0; i < 8; i++) {
    state[i] = cv[i];
  }
#pragma unroll
  for (auto i = 0; i < 4; i++) {
    state[8 + i] = IV[i];
  }

  state[12] = (uint32_t)counter;
  state[13] = (uint32_t)(counter >> 32);
  state[14] = (uint32_t)block_len;
  state[15] = (uint32_t)flags;

  round_fn(state, &block_words[0], 0);
  round_fn(state, &block_words[0], 1);
  round_fn(state, &block_words[0], 2);
  round_fn(state, &block_words[1], 3);
  round_fn(state, &block_words[0], 4);
  round_fn(state, &block_words[0], 5);
  round_fn(state, &block_words[0], 6);

  cv[0] = state[0] ^ state[8];
  cv[1] = state[1] ^ state[9];
  cv[2] = state[2] ^ state[10];
  cv[3] = state[3] ^ state[11];
  cv[4] = state[4] ^ state[12];
  cv[5] = state[5] ^ state[13];
  cv[6] = state[6] ^ state[14];
  cv[7] = state[7] ^ state[15];
}

__global__ void
blake3_compress_xof_cuda_kernel(uint32_t *cv, const uint8_t *block,
                                const uint8_t *out, uint8_t block_len,
                                uint64_t counter, uint8_t flags) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t state[16];       // v0 - v15
  uint32_t block_words[16]; // m0 - m15
#pragma unroll
  for (auto i = 0; i < 16; i++) {
    block_words[i] = *((uint32_t *)(&block + i * 4));
  }

#pragma unroll
  for (auto i = 0; i < 8; i++) {
    state[i] = cv[i];
  }
#pragma unroll
  for (auto i = 0; i < 4; i++) {
    state[8 + i] = IV[i];
  }

  state[12] = (uint32_t)counter;
  state[13] = (uint32_t)(counter >> 32);
  state[14] = (uint32_t)block_len;
  state[15] = (uint32_t)flags;

  round_fn(state, &block_words[0], 0);
  round_fn(state, &block_words[0], 1);
  round_fn(state, &block_words[0], 2);
  round_fn(state, &block_words[1], 3);
  round_fn(state, &block_words[0], 4);
  round_fn(state, &block_words[0], 5);
  round_fn(state, &block_words[0], 6);

  store32((uint8_t *)&out[0 * 4], state[0] ^ state[8]);
  store32((uint8_t *)&out[1 * 4], state[1] ^ state[9]);
  store32((uint8_t *)&out[2 * 4], state[2] ^ state[10]);
  store32((uint8_t *)&out[3 * 4], state[3] ^ state[11]);
  store32((uint8_t *)&out[4 * 4], state[4] ^ state[12]);
  store32((uint8_t *)&out[5 * 4], state[5] ^ state[13]);
  store32((uint8_t *)&out[6 * 4], state[6] ^ state[14]);
  store32((uint8_t *)&out[7 * 4], state[7] ^ state[15]);
  store32((uint8_t *)&out[8 * 4], state[8] ^ cv[0]);
  store32((uint8_t *)&out[9 * 4], state[9] ^ cv[1]);
  store32((uint8_t *)&out[10 * 4], state[10] ^ cv[2]);
  store32((uint8_t *)&out[11 * 4], state[11] ^ cv[3]);
  store32((uint8_t *)&out[12 * 4], state[12] ^ cv[4]);
  store32((uint8_t *)&out[13 * 4], state[13] ^ cv[5]);
  store32((uint8_t *)&out[14 * 4], state[14] ^ cv[6]);
  store32((uint8_t *)&out[15 * 4], state[15] ^ cv[7]);
}

#define SIMT_DEGREE 1024

__global__ void
blake3_hash_many_cuda_kernel(const uint8_t *d_inputs, size_t blocks,
                             const uint32_t key[8], uint64_t counter,
                             bool increment_counter, uint8_t flags,
                             uint8_t flags_start, uint8_t flags_end,
                             uint32_t *d_out, uint32_t *d_states) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < blocks) {
    uint32_t *state = d_states + 16 * id;
    uint32_t *block_words =
        (uint32_t *)(d_inputs + id * blocks * BLAKE3_BLOCK_LEN);
    const uint32_t *cv = key;
    uint32_t *local_out = d_out + id * 8;
    uint8_t block_flags = flags;
    if (id == 0) {
      block_flags |= flags_start;
    }
    if (id == blocks - 1) {
      block_flags |= flags_end;
    }
    uint32_t local_counter = increment_counter ? counter + id : counter;
#pragma unroll
    for (auto i = 0; i < 16; i++) {
      block_words[i] = *((uint32_t *)(&d_inputs + i * 4));
    }

#pragma unroll
    for (auto i = 0; i < 8; i++) {
      state[i] = cv[i];
    }
#pragma unroll
    for (auto i = 0; i < 4; i++) {
      state[8 + i] = IV[i];
    }

    state[12] = (uint32_t)local_counter;
    state[13] = (uint32_t)(local_counter >> 32);
    state[14] = (uint32_t)BLAKE3_BLOCK_LEN;
    state[15] = (uint32_t)block_flags;

    round_fn(state, &block_words[0], 0);
    round_fn(state, &block_words[0], 1);
    round_fn(state, &block_words[0], 2);
    round_fn(state, &block_words[1], 3);
    round_fn(state, &block_words[0], 4);
    round_fn(state, &block_words[0], 5);
    round_fn(state, &block_words[0], 6);

    local_out[0] = state[0] ^ state[8];
    local_out[1] = state[1] ^ state[9];
    local_out[2] = state[2] ^ state[10];
    local_out[3] = state[3] ^ state[11];
    local_out[4] = state[4] ^ state[12];
    local_out[5] = state[5] ^ state[13];
    local_out[6] = state[6] ^ state[14];
    local_out[7] = state[7] ^ state[15];
  }
}

void blake3_hash_many_cuda(const uint8_t *const *inputs, size_t num_inputs,
                           size_t blocks, const uint32_t key[8],
                           uint64_t counter, bool increment_counter,
                           uint8_t flags, uint8_t flags_start,
                           uint8_t flags_end, uint8_t *out) {

  uint32_t *d_state, *d_in, *d_out, *d_cv;
  checkCudaErrors(cudaMalloc((void **)&d_state, 64 * blocks));
  checkCudaErrors(
      cudaMalloc((void **)&d_in, BLAKE3_BLOCK_LEN * blocks * num_inputs));
  checkCudaErrors(cudaMalloc((void **)&d_out, 64 * blocks));
  checkCudaErrors(cudaMalloc((void **)&d_cv, 32));

  cudaMemcpyAsync(d_cv, key, BLAKE3_KEY_LEN, cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(d_in, inputs, blocks * BLAKE3_BLOCK_LEN,
                  cudaMemcpyHostToDevice, 0);

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_cv, key, BLAKE3_KEY_LEN, cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(d_in, inputs, BLAKE3_BLOCK_LEN * blocks,
                  cudaMemcpyHostToDevice, 0);
  dim3 block(blocks / 1024, 1, 1);
  dim3 thread(1024, 1, 1);
  blake3_hash_many_cuda_kernel<<<block, thread, 0, 0>>>(
      (uint8_t *)d_in, blocks, d_cv, counter, increment_counter, flags,
      flags_start, flags_end, d_out, d_state);

  cudaMemcpyAsync(out, d_out, 64 * blocks, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFree(d_state));
  checkCudaErrors(cudaFree(d_cv));
  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_out));
}

void blake3_compress_in_place_cuda(uint32_t cv[8],
                                   const uint8_t block[BLAKE3_BLOCK_LEN],
                                   uint8_t block_len, uint64_t counter,
                                   uint8_t flags) {
  uint32_t *d_cv;
  uint8_t *d_in;
  checkCudaErrors(cudaMalloc((void **)&d_cv, BLAKE3_KEY_LEN));
  checkCudaErrors(cudaMalloc((void **)&d_in, block_len));
  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_cv, cv, BLAKE3_KEY_LEN, cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(d_in, block, block_len, cudaMemcpyHostToDevice, 0);
  blake3_compress_in_place_cuda_kernel<<<1, 1, 0, 0>>>(d_cv, d_in, block_len,
                                                       counter, flags);

  cudaMemcpyAsync(cv, d_cv, BLAKE3_KEY_LEN, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_cv));
}

void hash_one_cuda(const uint8_t *input, size_t blocks, const uint32_t key[8],
                   uint64_t counter, uint8_t flags, uint8_t flags_start,
                   uint8_t flags_end, uint8_t out[BLAKE3_OUT_LEN]) {
  uint32_t cv[8];
  memcpy(cv, key, BLAKE3_KEY_LEN);
  uint32_t *d_cv;
  uint8_t *d_in;
  checkCudaErrors(cudaMalloc((void **)&d_cv, BLAKE3_KEY_LEN));
  checkCudaErrors(cudaMalloc((void **)&d_in, BLAKE3_BLOCK_LEN));
  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_cv, key, BLAKE3_KEY_LEN, cudaMemcpyHostToDevice, 0);
  uint8_t block_flags = flags | flags_start;
  while (blocks > 0) {
    if (blocks == 1) {
      block_flags |= flags_end;
    }
    cudaMemcpyAsync(d_in, input, BLAKE3_BLOCK_LEN, cudaMemcpyHostToDevice, 0);
    blake3_compress_in_place_cuda_kernel<<<1, 1, 0, 0>>>(
        d_cv, d_in, BLAKE3_BLOCK_LEN, counter, block_flags);
    input = &input[BLAKE3_BLOCK_LEN];
    blocks -= 1;
    block_flags = flags;
  }

  cudaMemcpyAsync(cv, d_cv, BLAKE3_KEY_LEN, cudaMemcpyDeviceToHost, 0);
  store_cv_words(out, cv);
  cudaEventRecord(stop, 0);

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_cv));
}

void blake3_compress_xof_cuda(const uint32_t cv[8],
                              const uint8_t block[BLAKE3_BLOCK_LEN],
                              uint8_t block_len, uint64_t counter,
                              uint8_t flags, uint8_t out[64]) {
  uint32_t *d_cv;
  uint8_t *d_in, *d_out;
  checkCudaErrors(cudaMalloc((void **)&d_cv, BLAKE3_KEY_LEN));
  checkCudaErrors(cudaMalloc((void **)&d_in, block_len));
  checkCudaErrors(cudaMalloc((void **)&d_out, 64));

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_cv, cv, BLAKE3_KEY_LEN, cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(d_in, block, block_len, cudaMemcpyHostToDevice, 0);
  blake3_compress_xof_cuda_kernel<<<1, 1, 0, 0>>>(d_cv, d_in, d_out, block_len,
                                                  counter, flags);

  cudaMemcpyAsync(out, d_out, 64, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_cv));
}

int main() {}

#ifdef __cplusplus
}
#endif
