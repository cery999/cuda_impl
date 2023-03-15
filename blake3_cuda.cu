// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./blake3_header.cuh"
// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h> // helper functions for CUDA error checking and initialization
#include <helper_functions.h> // helper utility functions


#ifdef __cplusplus
extern "C" {
#endif

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
  state[d] = rotr32(state[d] ^ state[a], 16);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 12);
  state[a] = state[a] + state[b] + y;
  state[d] = rotr32(state[d] ^ state[a], 8);
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

__device__ inline uint32_t load32(const void *src) {
  const uint8_t *p = (const uint8_t *)src;
  return ((uint32_t)(p[0]) << 0) | ((uint32_t)(p[1]) << 8) |
         ((uint32_t)(p[2]) << 16) | ((uint32_t)(p[3]) << 24);
}

__device__ inline uint32_t counter_low(uint64_t counter) {
  return (uint32_t)counter;
}
__device__ inline uint32_t counter_high(uint64_t counter) {
  return (uint32_t)(counter >> 32);
}

__device__ void compress_pre(uint32_t state[16], const uint32_t cv[8],
                             const uint8_t block[BLAKE3_BLOCK_LEN],
                             uint8_t block_len, uint64_t counter,
                             uint8_t flags) {
  uint32_t block_words[16];
  block_words[0] = load32(block + 4 * 0);
  block_words[1] = load32(block + 4 * 1);
  block_words[2] = load32(block + 4 * 2);
  block_words[3] = load32(block + 4 * 3);
  block_words[4] = load32(block + 4 * 4);
  block_words[5] = load32(block + 4 * 5);
  block_words[6] = load32(block + 4 * 6);
  block_words[7] = load32(block + 4 * 7);
  block_words[8] = load32(block + 4 * 8);
  block_words[9] = load32(block + 4 * 9);
  block_words[10] = load32(block + 4 * 10);
  block_words[11] = load32(block + 4 * 11);
  block_words[12] = load32(block + 4 * 12);
  block_words[13] = load32(block + 4 * 13);
  block_words[14] = load32(block + 4 * 14);
  block_words[15] = load32(block + 4 * 15);

  state[0] = cv[0];
  state[1] = cv[1];
  state[2] = cv[2];
  state[3] = cv[3];
  state[4] = cv[4];
  state[5] = cv[5];
  state[6] = cv[6];
  state[7] = cv[7];
  state[8] = IV[0];
  state[9] = IV[1];
  state[10] = IV[2];
  state[11] = IV[3];
  state[12] = counter_low(counter);
  state[13] = counter_high(counter);
  state[14] = (uint32_t)block_len;
  state[15] = (uint32_t)flags;

  round_fn(state, &block_words[0], 0);
  round_fn(state, &block_words[0], 1);
  round_fn(state, &block_words[0], 2);
  round_fn(state, &block_words[0], 3);
  round_fn(state, &block_words[0], 4);
  round_fn(state, &block_words[0], 5);
  round_fn(state, &block_words[0], 6);
}

__device__ void blake3_compress_in_place_cuda_kernel(
    uint32_t cv[8], const uint8_t block[BLAKE3_BLOCK_LEN], uint8_t block_len,
    uint64_t counter, uint8_t flags) {
  uint32_t state[16];
  compress_pre(state, cv, block, block_len, counter, flags);
  cv[0] = state[0] ^ state[8];
  cv[1] = state[1] ^ state[9];
  cv[2] = state[2] ^ state[10];
  cv[3] = state[3] ^ state[11];
  cv[4] = state[4] ^ state[12];
  cv[5] = state[5] ^ state[13];
  cv[6] = state[6] ^ state[14];
  cv[7] = state[7] ^ state[15];
}

__global__ void blake3_compress_in_place_cuda_global_kernel(
    uint32_t cv[8], const uint8_t block[BLAKE3_BLOCK_LEN], uint8_t block_len,
    uint64_t counter, uint8_t flags) {
  blake3_compress_in_place_cuda_kernel(cv, block, block_len, counter, flags);
}

void blake3_compress_in_place_cuda(uint32_t cv[8],
                                   const uint8_t block[BLAKE3_BLOCK_LEN],
                                   uint8_t block_len, uint64_t counter,
                                   uint8_t flags) {

  checkCudaErrors(cudaProfilerStart());
  uint32_t *d_cv;
  uint8_t *d_block;
  checkCudaErrors(cudaMalloc((void **)&d_cv, 8 * sizeof(uint32_t)));
  checkCudaErrors(cudaMalloc((void **)&d_block, BLAKE3_BLOCK_LEN));

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_cv, cv, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(d_block, block, BLAKE3_BLOCK_LEN, cudaMemcpyHostToDevice, 0);

  blake3_compress_in_place_cuda_global_kernel<<<1, 1, 0, 0>>>(
      d_cv, d_block, block_len, counter, flags);
  cudaMemcpyAsync(cv, d_cv, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost, 0);

  cudaEventRecord(stop, 0);

  // destroy stream
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  checkCudaErrors(cudaFree(d_cv));
  checkCudaErrors(cudaFree(d_block));

  checkCudaErrors(cudaProfilerStop());
}

__global__ void
blake3_hash_many_cuda_kernel(const uint8_t *d_inputs, uint32_t *d_cv,
                             uint8_t *d_out, size_t num_inputs, size_t blocks,
                             const uint32_t d_key[8], uint64_t counter,
                             bool increment_counter, uint8_t flags,
                             uint8_t flags_start, uint8_t flags_end) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < num_inputs) {
    uint32_t *cv = d_cv + id * 8;
    uint8_t *out = d_out + id * 32;
    const uint8_t *input = d_inputs + id * blocks * BLAKE3_BLOCK_LEN;
    for (auto i = 0; i < 8; i++) {
      cv[i] = d_key[i];
    }
    if (increment_counter) {
      counter += id;
    }
    uint8_t block_flags = flags | flags_start;
    while (blocks > 0) {
      if (blocks == 1) {
        block_flags |= flags_end;
      }
      blake3_compress_in_place_cuda_kernel(cv, input, BLAKE3_BLOCK_LEN, counter,
                                           block_flags);
      input = &input[BLAKE3_BLOCK_LEN];
      blocks -= 1;
      block_flags = flags;
    }
    store_cv_words(out, cv);
  }
}

void blake3_hash_many_cuda(const uint8_t *const *inputs, size_t num_inputs,
                           size_t blocks, const uint32_t key[8],
                           uint64_t counter, bool increment_counter,
                           uint8_t flags, uint8_t flags_start,
                           uint8_t flags_end, uint8_t *out) {
  uint32_t *d_cv, *d_key;
  uint8_t *d_out, *d_in;
  checkCudaErrors(
      cudaMalloc((void **)&d_in, BLAKE3_BLOCK_LEN * blocks * num_inputs));
  checkCudaErrors(cudaMalloc((void **)&d_out, 32 * num_inputs));
  checkCudaErrors(cudaMalloc((void **)&d_cv, 32 * num_inputs));
  checkCudaErrors(cudaMalloc((void **)&d_key, 32));

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  cudaEventRecord(start, 0);
  auto offset = 0;
  for (auto i = 0; i < num_inputs; i++) {
    cudaMemcpyAsync(d_in + offset, inputs[i], blocks * BLAKE3_BLOCK_LEN,
                    cudaMemcpyHostToDevice, 0);
    offset += blocks * BLAKE3_BLOCK_LEN;
  }
  cudaMemcpyAsync(d_key, key, BLAKE3_KEY_LEN, cudaMemcpyHostToDevice, 0);

  dim3 block, thread;
  if (num_inputs < 1024) {
    block = dim3(1, 1, 1);
    thread = dim3(num_inputs, 1, 1);
  } else {
    block = dim3(ceil(num_inputs * 1.0 / 1024), 1, 1);
    thread = dim3(1024, 1, 1);
  }

  blake3_hash_many_cuda_kernel<<<block, thread, 0, 0>>>(
      d_in, d_cv, d_out, num_inputs, blocks, d_key, counter, increment_counter,
      flags, flags_start, flags_end);
  cudaMemcpyAsync(out, d_out, 32 * num_inputs, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);

  // destroy stream
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  // free
  checkCudaErrors(cudaFree(d_cv));
  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_out));
}

__global__ void blake3_compress_xof_cuda_kernel(
    const uint32_t cv[8], const uint8_t block[BLAKE3_BLOCK_LEN],
    uint8_t block_len, uint64_t counter, uint8_t flags, uint8_t out[64]) {
  uint32_t state[16];
  compress_pre(state, cv, block, block_len, counter, flags);

  store32(&out[0 * 4], state[0] ^ state[8]);
  store32(&out[1 * 4], state[1] ^ state[9]);
  store32(&out[2 * 4], state[2] ^ state[10]);
  store32(&out[3 * 4], state[3] ^ state[11]);
  store32(&out[4 * 4], state[4] ^ state[12]);
  store32(&out[5 * 4], state[5] ^ state[13]);
  store32(&out[6 * 4], state[6] ^ state[14]);
  store32(&out[7 * 4], state[7] ^ state[15]);
  store32(&out[8 * 4], state[8] ^ cv[0]);
  store32(&out[9 * 4], state[9] ^ cv[1]);
  store32(&out[10 * 4], state[10] ^ cv[2]);
  store32(&out[11 * 4], state[11] ^ cv[3]);
  store32(&out[12 * 4], state[12] ^ cv[4]);
  store32(&out[13 * 4], state[13] ^ cv[5]);
  store32(&out[14 * 4], state[14] ^ cv[6]);
  store32(&out[15 * 4], state[15] ^ cv[7]);
}

void blake3_compress_xof_cuda(const uint32_t cv[8],
                              const uint8_t block[BLAKE3_BLOCK_LEN],
                              uint8_t block_len, uint64_t counter,
                              uint8_t flags, uint8_t out[64]) {
  uint32_t *d_cv;
  uint8_t *d_out, *d_block;
  checkCudaErrors(cudaMalloc((void **)&d_cv, 8 * sizeof(uint32_t)));
  checkCudaErrors(cudaMalloc((void **)&d_out, 64 * sizeof(uint8_t)));
  checkCudaErrors(cudaMalloc((void **)&d_block, BLAKE3_BLOCK_LEN));

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_cv, cv, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(d_block, block, BLAKE3_BLOCK_LEN, cudaMemcpyHostToDevice, 0);
  blake3_compress_xof_cuda_kernel<<<1, 1, 0, 0>>>(d_cv, d_block, block_len,
                                                  counter, flags, d_out);
  cudaMemcpyAsync(out, d_out, 64 * sizeof(uint8_t), cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);

  // destroy stream
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  checkCudaErrors(cudaFree(d_cv));
  checkCudaErrors(cudaFree(d_block));
  checkCudaErrors(cudaFree(d_out));
}

void test1() {
  uint8_t **inputs = (uint8_t **)malloc(31 * sizeof(uint8_t **));
  for (auto i = 0; i < 31; i++) {
    inputs[i] = (uint8_t *)malloc(1024 * sizeof(uint8_t));
  }
  uint32_t key[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t out[32 * 31];

  blake3_hash_many_cuda(inputs, 31, 16, key, 0, true, 1, 2, 4, out);
}

extern "C" void pre_allocate() {
  checkCudaErrors(
      cudaMalloc((void **)&pined_d_hasher, 102400 * sizeof(blake3_hasher)));
}

extern "C" void post_free() { checkCudaErrors(cudaFree(pined_d_hasher)); }

void test_init() {
  checkCudaErrors(cudaProfilerStart());

  blake3_hasher_init<<<2, 1024, 0, 0>>>(pined_d_hasher);
  checkCudaErrors(cudaProfilerStop());
}

/* int main() { */
/*   pre_allocate(); */
/*   /1* test1(); *1/ */
/*   test_init(); */
/*   post_free(); */
/* } */


__device__ void chunk_state_init(thread_block g, blake3_hasher *d_hash,
                                 const uint32_t key[8], uint8_t flags) {
  auto group_dim = g.group_dim();
  auto group_idx = g.group_index();
  auto idx = group_dim.x * group_idx.x + g.thread_rank();
  blake3_chunk_state *chunk_state = &d_hash[idx].chunk;

  memcpy(chunk_state->cv, key, BLAKE3_KEY_LEN);
  chunk_state->chunk_counter = 0;
  memset(chunk_state->buf, 0, BLAKE3_BLOCK_LEN);
  chunk_state->buf_len = 0;
  chunk_state->blocks_compressed = 0;
  chunk_state->flags = flags;
}

__device__ void hasher_init_base(thread_block g, blake3_hasher *d_hash,
                                 const uint32_t key[8], uint8_t flags) {
  auto group_dim = g.group_dim();
  auto group_idx = g.group_index();
  auto idx = group_dim.x * group_idx.x + g.thread_rank();
#pragma unroll
  for (auto i = 0; i < 8; i++) {
    d_hash[idx].key[i] = key[i];
  }
  chunk_state_init(g, d_hash, key, flags);
  d_hash[idx].cv_stack_len = 0;
}

__global__ void blake3_hasher_init(blake3_hasher *d_hash) {
  thread_block g = this_thread_block();
  coalesced_group wrap = coalesced_threads();
  auto lane = g.thread_rank();
  auto thread_id_in_wrap = wrap.thread_rank();
  uint32_t k[8];
  if (thread_id_in_wrap == 0) {
#pragma unroll
    for (auto i = 0; i < 8; i++) {
      /* printf("%d\n", g.thread_rank()); */
      k[i] = IV[i];
    }
  }
  for (auto i = 0; i < 8; i++) {
    k[i] = wrap.shfl(k[i], 0);
  }
  __syncwarp();
  hasher_init_base(g, d_hash, k, 0);
}



#ifdef __cplusplus
}
#endif
