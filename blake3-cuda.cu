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

__global__ void blake3_compress_in_place_portable(
    uint32_t* cv, const uint8_t* block, uint8_t block_len,
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
  round_fn(state, &block_words[0], 3);
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

inline void store32(void *dst, uint32_t w) {
  uint8_t *p = (uint8_t *)dst;
  p[0] = (uint8_t)(w >> 0);
  p[1] = (uint8_t)(w >> 8);
  p[2] = (uint8_t)(w >> 16);
  p[3] = (uint8_t)(w >> 24);
}

inline void store_cv_words(uint8_t bytes_out[32], uint32_t cv_words[8]) {
  store32(&bytes_out[0 * 4], cv_words[0]);
  store32(&bytes_out[1 * 4], cv_words[1]);
  store32(&bytes_out[2 * 4], cv_words[2]);
  store32(&bytes_out[3 * 4], cv_words[3]);
  store32(&bytes_out[4 * 4], cv_words[4]);
  store32(&bytes_out[5 * 4], cv_words[5]);
  store32(&bytes_out[6 * 4], cv_words[6]);
  store32(&bytes_out[7 * 4], cv_words[7]);
}

void hash_one_cuda(const uint8_t *input, size_t blocks,
                       const uint32_t key[8], uint64_t counter, uint8_t flags,
                       uint8_t flags_start, uint8_t flags_end,
                       uint8_t out[BLAKE3_OUT_LEN]) {
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
    blake3_compress_in_place_portable<<<1, 1, 0, 0>>>(d_cv, d_in, BLAKE3_BLOCK_LEN,
                                                counter, block_flags);
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

int main(int argc, char *argv[]) {

}

#ifdef __cplusplus
}
#endif
