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

#define MAX_SIMD_DEGREE_OR_2 1024
#define MAX_SIMD_DEGREE 1024

__device__ void chunk_state_reset(blake3_chunk_state *self,
                                  const uint32_t key[8],
                                  uint64_t chunk_counter);
__device__ uint8_t chunk_state_maybe_start_flag(const blake3_chunk_state *self);
__global__ void blake3_hasher_reset(blake3_hasher *d_hash, size_t N);
__global__ void blake3_hasher_init(blake3_hasher *d_hash, size_t N);

__device__ unsigned int highest_one(uint64_t x) { return 63 ^ __clzll(x); }
__device__ uint64_t round_down_to_power_of_2(uint64_t x) {
  return 1ULL << highest_one(x | 1);
}

__device__ uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
}

__device__ void store32(void *dst, uint32_t w) {
  uint8_t *p = (uint8_t *)dst;
  p[0] = (uint8_t)(w >> 0);
  p[1] = (uint8_t)(w >> 8);
  p[2] = (uint8_t)(w >> 16);
  p[3] = (uint8_t)(w >> 24);
}

__device__ void store_cv_words(uint8_t bytes_out[32], uint32_t cv_words[8]) {
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

__device__ void hash_one_cuda(const uint8_t *input, size_t blocks,
                              const uint32_t key[8], uint64_t counter,
                              uint8_t flags, uint8_t flags_start,
                              uint8_t flags_end, uint8_t out[BLAKE3_OUT_LEN]) {
  uint32_t cv[8];
  memcpy(cv, key, BLAKE3_KEY_LEN);
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

__global__ void blake3_hash_many_cuda_kernel(
    const uint8_t *inputs, size_t num_inputs, size_t blocks,
    const uint32_t key[8], uint64_t counter, bool increment_counter,
    uint8_t flags, uint8_t flags_start, uint8_t flags_end, uint8_t *out) {

  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_inputs) {
    if (increment_counter) {
      counter += idx;
    }
    auto thread_inputs = inputs + idx * BLAKE3_CHUNK_LEN;
    hash_one_cuda(thread_inputs, blocks, key, counter, flags, flags_start,
                  flags_end, out + idx * BLAKE3_OUT_LEN);
  }
}

void blake3_hash_many_cuda(const uint8_t *const *inputs, size_t num_inputs,
                           size_t blocks, const uint32_t key[8],
                           uint64_t counter, bool increment_counter,
                           uint8_t flags, uint8_t flags_start,
                           uint8_t flags_end, uint8_t *out) {
  uint32_t *d_key;
  uint8_t *d_out, *d_in;
  size_t pitch;
  cudaMallocPitch(&d_in, &pitch, BLAKE3_CHUNK_LEN, num_inputs);
  checkCudaErrors(
      cudaMalloc((void **)&d_in, BLAKE3_BLOCK_LEN * blocks * num_inputs));
  checkCudaErrors(cudaMalloc((void **)&d_out, 32 * num_inputs));
  checkCudaErrors(cudaMalloc((void **)&d_key, 32));

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  cudaEventRecord(start, 0);

  cudaMemcpy2DAsync(d_in, pitch, inputs, pitch, BLAKE3_CHUNK_LEN, num_inputs,
                    cudaMemcpyHostToDevice);
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
      d_in, num_inputs, blocks, d_key, counter, increment_counter, flags,
      flags_start, flags_end, d_out);
  cudaMemcpyAsync(out, d_out, 32 * num_inputs, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);

  // destroy stream
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  // free
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

extern "C" void blake3_hasher_init_cuda(blake3_hasher *data, size_t N) {
  if (N < 1024) {
    blake3_hasher_init<<<1, N, 0, 0>>>(pined_d_hasher, N);
  } else {
    blake3_hasher_init<<<ceil(N / 1024), 1024, 0, 0>>>(pined_d_hasher, N);
  }
}

__global__ void blake3_hasher_reset(blake3_hasher *d_hash, size_t N) {
  thread_block g = this_thread_block();
  auto group_dim = g.group_dim();
  auto group_idx = g.group_index();
  auto idx = group_idx.x * group_dim.x + g.thread_rank();
  if (idx < N) {
    blake3_hasher *self = &d_hash[idx];
    chunk_state_reset(&self->chunk, self->key, 0);
    self->cv_stack_len = 0;
  }
}

extern "C" void blake3_hasher_reset_cuda(blake3_hasher *data, size_t N) {
  if (N < 1024) {
    blake3_hasher_reset<<<1, N, 0, 0>>>(pined_d_hasher, N);
  } else {
    blake3_hasher_reset<<<ceil(N / 1024), 1024, 0, 0>>>(pined_d_hasher, N);
  }
}

/* int main() { */
/*   pre_allocate(); */
/*   /1* test1(); *1/ */
/*   post_free(); */
/* } */

__device__ void chunk_state_init(blake3_chunk_state *self,
                                 const uint32_t key[8], uint8_t flags) {
  memcpy(self->cv, key, BLAKE3_KEY_LEN);
  self->chunk_counter = 0;
  memset(self->buf, 0, BLAKE3_BLOCK_LEN);
  self->buf_len = 0;
  self->blocks_compressed = 0;
  self->flags = flags;
}

__device__ void hasher_init_base(blake3_hasher *self, const uint32_t key[8],
                                 uint8_t flags) {
  memcpy(self->key, key, BLAKE3_KEY_LEN);
  chunk_state_init(&self->chunk, key, flags);
  self->cv_stack_len = 0;
}

__global__ void blake3_hasher_init(blake3_hasher *d_hash, size_t N) {
  thread_block g = this_thread_block();
  auto group_dim = g.group_dim();
  auto group_idx = g.group_index();
  auto idx = group_idx.x * group_dim.x + g.thread_rank();

  coalesced_group wrap = coalesced_threads();
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
  if (idx < N) {
    blake3_hasher *self = &d_hash[idx];
    hasher_init_base(self, IV, 0);
  }
}

__device__ void chunk_state_reset(blake3_chunk_state *self,
                                  const uint32_t key[8],
                                  uint64_t chunk_counter) {
  memcpy(self->cv, key, BLAKE3_KEY_LEN);
  self->chunk_counter = chunk_counter;
  self->blocks_compressed = 0;
  memset(self->buf, 0, BLAKE3_BLOCK_LEN);
  self->buf_len = 0;
}

__device__ size_t chunk_state_len(thread_block g, const blake3_hasher *d_hash) {
  auto group_dim = g.group_dim();
  auto group_idx = g.group_index();
  auto idx = group_dim.x * group_idx.x + g.thread_rank();
  const blake3_chunk_state *self = &d_hash[idx].chunk;
  return (BLAKE3_BLOCK_LEN * (size_t)self->blocks_compressed) +
         ((size_t)self->buf_len);
}

__device__ size_t chunk_state_fill_buf(blake3_chunk_state *self,
                                       const uint8_t *input, size_t input_len) {
  size_t take = BLAKE3_BLOCK_LEN - ((size_t)self->buf_len);
  if (take > input_len) {
    take = input_len;
  }
  uint8_t *dest = self->buf + ((size_t)self->buf_len);
  memcpy(dest, input, take);
  self->buf_len += (uint8_t)take;
  return take;
}

__device__ void chunk_state_update(blake3_chunk_state *self,
                                   const uint8_t *input, size_t input_len) {
  if (self->buf_len > 0) {
    size_t take = chunk_state_fill_buf(self, input, input_len);
    input += take;
    input_len -= take;
    if (input_len > 0) {
      blake3_compress_in_place_cuda_kernel(
          self->cv, self->buf, BLAKE3_BLOCK_LEN, self->chunk_counter,
          self->flags | chunk_state_maybe_start_flag(self));
      self->blocks_compressed += 1;
      self->buf_len = 0;
      memset(self->buf, 0, BLAKE3_BLOCK_LEN);
    }
  }

  while (input_len > BLAKE3_BLOCK_LEN) {
    blake3_compress_in_place_cuda_kernel(
        self->cv, input, BLAKE3_BLOCK_LEN, self->chunk_counter,
        self->flags | chunk_state_maybe_start_flag(self));
    self->blocks_compressed += 1;
    input += BLAKE3_BLOCK_LEN;
    input_len -= BLAKE3_BLOCK_LEN;
  }

  size_t take = chunk_state_fill_buf(self, input, input_len);
  input += take;
  input_len -= take;
}

__device__ uint8_t
chunk_state_maybe_start_flag(const blake3_chunk_state *self) {
  if (self->blocks_compressed == 0) {
    return CHUNK_START;
  } else {
    return 0;
  }
}

__device__ output_t make_output(const uint32_t input_cv[8],
                                const uint8_t block[BLAKE3_BLOCK_LEN],
                                uint8_t block_len, uint64_t counter,
                                uint8_t flags) {
  output_t ret;
  memcpy(ret.input_cv, input_cv, 32);
  memcpy(ret.block, block, BLAKE3_BLOCK_LEN);
  ret.block_len = block_len;
  ret.counter = counter;
  ret.flags = flags;
  return ret;
}

__device__ output_t chunk_state_output(const blake3_chunk_state *self) {
  uint8_t block_flags =
      self->flags | chunk_state_maybe_start_flag(self) | CHUNK_END;
  return make_output(self->cv, self->buf, self->buf_len, self->chunk_counter,
                     block_flags);
}

__device__ void output_chaining_value(const output_t *self, uint8_t cv[32]) {
  uint32_t cv_words[8];
  memcpy(cv_words, self->input_cv, 32);
  blake3_compress_in_place_cuda_kernel(cv_words, self->block, self->block_len,
                                       self->counter, self->flags);
  store_cv_words(cv, cv_words);
}

__device__ output_t parent_output(const uint8_t block[BLAKE3_BLOCK_LEN],
                                  const uint32_t key[8], uint8_t flags) {
  return make_output(key, block, BLAKE3_BLOCK_LEN, 0, flags | PARENT);
}

__device__ void hasher_merge_cv_stack(blake3_hasher *self, uint64_t total_len) {
  size_t post_merge_stack_len = (size_t)__popcll(total_len);
  while (self->cv_stack_len > post_merge_stack_len) {
    uint8_t *parent_node =
        &self->cv_stack[(self->cv_stack_len - 2) * BLAKE3_OUT_LEN];
    output_t output = parent_output(parent_node, self->key, self->chunk.flags);
    output_chaining_value(&output, parent_node);
    self->cv_stack_len -= 1;
  }
}
__device__ void hasher_push_cv(blake3_hasher *self,
                               uint8_t new_cv[BLAKE3_OUT_LEN],
                               uint64_t chunk_counter) {
  hasher_merge_cv_stack(self, chunk_counter);
  memcpy(&self->cv_stack[self->cv_stack_len * BLAKE3_OUT_LEN], new_cv,
         BLAKE3_OUT_LEN);
  self->cv_stack_len += 1;
}

__device__ size_t compress_chunks_parallel(const uint8_t *input,
                                           size_t input_len,
                                           const uint32_t key[8],
                                           uint64_t chunk_counter,
                                           uint8_t flags, uint8_t *out) {
#if defined(BLAKE3_TESTING)
  assert(0 < input_len);
  // shared max 48kB
  assert(input_len <= MAX_SIMD_DEGREE * BLAKE3_CHUNK_LEN);
#endif
  auto num_inputs = input_len / BLAKE3_CHUNK_LEN;
  if (num_inputs > 0) {
    blake3_hash_many_cuda_kernel<<<num_inputs, MAX_SIMD_DEGREE>>>(
        input, num_inputs, BLAKE3_CHUNK_LEN / BLAKE3_BLOCK_LEN, key,
        chunk_counter, true, flags, CHUNK_START, CHUNK_END, out);
  } else {
    blake3_hash_many_cuda_kernel<<<1, num_inputs>>>(
        input, num_inputs, BLAKE3_CHUNK_LEN / BLAKE3_BLOCK_LEN, key,
        chunk_counter, true, flags, CHUNK_START, CHUNK_END, out);
  }

  auto num_inputs_left = input_len % BLAKE3_CHUNK_LEN;
  if (num_inputs_left) {
    uint64_t counter = chunk_counter + (uint64_t)num_inputs;
    blake3_chunk_state chunk_state;
    chunk_state_init(&chunk_state, key, flags);
    chunk_state.chunk_counter = counter;
    chunk_state_update(&chunk_state, &input[num_inputs * BLAKE3_CHUNK_LEN],
                       num_inputs_left);
    output_t output = chunk_state_output(&chunk_state);
    output_chaining_value(&output, &out[num_inputs * BLAKE3_OUT_LEN]);
    return num_inputs + 1;
  } else {
    return num_inputs;
  }
}

__device__ size_t left_len(size_t content_len) {
  // Subtract 1 to reserve at least one byte for the right side. content_len
  // should always be greater than BLAKE3_CHUNK_LEN.
  size_t full_chunks = (content_len - 1) / BLAKE3_CHUNK_LEN;
  return round_down_to_power_of_2(full_chunks) * BLAKE3_CHUNK_LEN;
}

__device__ size_t compress_parents_parallel(
    const uint8_t *child_chaining_values, size_t num_chaining_values,
    const uint32_t key[8], uint8_t flags, uint8_t *out) {
#if defined(BLAKE3_TESTING)
  assert(2 <= num_chaining_values);
  assert(num_chaining_values <= 2 * MAX_SIMD_DEGREE_OR_2);
#endif

  const uint8_t *parents_array[MAX_SIMD_DEGREE_OR_2];
  size_t parents_array_len = 0;
  while (num_chaining_values - (2 * parents_array_len) >= 2) {
    parents_array[parents_array_len] =
        &child_chaining_values[2 * parents_array_len * BLAKE3_OUT_LEN];
    parents_array_len += 1;
  }

  blake3_hash_many_cuda_kernel<<<1, parents_array_len>>>(
      child_chaining_values, parents_array_len, 1, key,
      0, // Parents always use counter 0.
      false, flags | PARENT,
      0, // Parents have no start flags.
      0, // Parents have no end flags.
      out);

  // If there's an odd child left over, it becomes an output.
  if (num_chaining_values > 2 * parents_array_len) {
    memcpy(&out[parents_array_len * BLAKE3_OUT_LEN],
           &child_chaining_values[2 * parents_array_len * BLAKE3_OUT_LEN],
           BLAKE3_OUT_LEN);
    return parents_array_len + 1;
  } else {
    return parents_array_len;
  }
}

__device__ size_t blake3_compress_subtree_wide(const uint8_t *input,
                                               size_t input_len,
                                               const uint32_t key[8],
                                               uint64_t chunk_counter,
                                               uint8_t flags, uint8_t *out) {
  // Note that the single chunk case does *not* bump the SIMD degree up to 2
  // when it is 1. If this implementation adds multi-threading in the future,
  // this gives us the option of multi-threading even the 2-chunk case, which
  // can help performance on smaller platforms.
  if (input_len <= 1024 * BLAKE3_CHUNK_LEN) {
    return compress_chunks_parallel(input, input_len, key, chunk_counter, flags,
                                    out);
  }

  // With more than simd_degree chunks, we need to recurse. Start by dividing
  // the input into left and right subtrees. (Note that this is only optimal
  // as long as the SIMD degree is a power of 2. If we ever get a SIMD degree
  // of 3 or something, we'll need a more complicated strategy.)
  size_t left_input_len = left_len(input_len);
  size_t right_input_len = input_len - left_input_len;
  const uint8_t *right_input = &input[left_input_len];
  uint64_t right_chunk_counter =
      chunk_counter + (uint64_t)(left_input_len / BLAKE3_CHUNK_LEN);

  // Make space for the child outputs. Here we use MAX_SIMD_DEGREE_OR_2 to
  // account for the special case of returning 2 outputs when the SIMD degree
  // is 1.
  uint8_t cv_array[2 * MAX_SIMD_DEGREE_OR_2 * BLAKE3_OUT_LEN];
  size_t degree = MAX_SIMD_DEGREE;
  if (left_input_len > BLAKE3_CHUNK_LEN && degree == 1) {
    // The special case: We always use a degree of at least two, to make
    // sure there are two outputs. Except, as noted above, at the chunk
    // level, where we allow degree=1. (Note that the 1-chunk-input case is
    // a different codepath.)
    degree = 2;
  }
  uint8_t *right_cvs = &cv_array[degree * BLAKE3_OUT_LEN];

  // Recurse! If this implementation adds multi-threading support in the
  // future, this is where it will go.
  size_t left_n = blake3_compress_subtree_wide(input, left_input_len, key,
                                               chunk_counter, flags, cv_array);
  size_t right_n = blake3_compress_subtree_wide(
      right_input, right_input_len, key, right_chunk_counter, flags, right_cvs);

  // The special case again. If simd_degree=1, then we'll have left_n=1 and
  // right_n=1. Rather than compressing them into a single output, return
  // them directly, to make sure we always have at least two outputs.
  if (left_n == 1) {
    memcpy(out, cv_array, 2 * BLAKE3_OUT_LEN);
    return 2;
  }

  // Otherwise, do one layer of parent node compression.
  size_t num_chaining_values = left_n + right_n;
  return compress_parents_parallel(cv_array, num_chaining_values, key, flags,
                                   out);
}

__device__ void compress_subtree_to_parent_node(
    const uint8_t *input, size_t input_len, const uint32_t key[8],
    uint64_t chunk_counter, uint8_t flags, uint8_t out[2 * BLAKE3_OUT_LEN]) {
#if defined(BLAKE3_TESTING)
  assert(input_len > BLAKE3_CHUNK_LEN);
#endif

  uint8_t cv_array[MAX_SIMD_DEGREE_OR_2 * BLAKE3_OUT_LEN];
  size_t num_cvs = blake3_compress_subtree_wide(input, input_len, key,
                                                chunk_counter, flags, cv_array);
  assert(num_cvs <= MAX_SIMD_DEGREE_OR_2);

  // If MAX_SIMD_DEGREE is greater than 2 and there's enough input,
  // compress_subtree_wide() returns more than 2 chaining values. Condense
  // them into 2 by forming parent nodes repeatedly.
  uint8_t out_array[MAX_SIMD_DEGREE_OR_2 * BLAKE3_OUT_LEN / 2];
  // The second half of this loop condition is always true, and we just
  // asserted it above. But GCC can't tell that it's always true, and if NDEBUG
  // is set on platforms where MAX_SIMD_DEGREE_OR_2 == 2, GCC emits spurious
  // warnings here. GCC 8.5 is particularly sensitive, so if you're changing
  // this code, test it against that version.
  while (num_cvs > 2 && num_cvs <= MAX_SIMD_DEGREE_OR_2) {
    num_cvs =
        compress_parents_parallel(cv_array, num_cvs, key, flags, out_array);
    memcpy(cv_array, out_array, num_cvs * BLAKE3_OUT_LEN);
  }
  memcpy(out, cv_array, 2 * BLAKE3_OUT_LEN);
}

__global__ void blake3_hasher_update(blake3_hasher *d_hash, const void *d_input,
                                     size_t *input_lens,
                                     size_t *input_partial_sum_lens, size_t N) {
  // Explicitly checking for zero avoids causing UB by passing a null pointer
  // to memcpy. This comes up in practice with things like:
  //   std::vector<uint8_t> v;
  //   blake3_hasher_update(&hasher, v.data(), v.size());
  thread_block g = this_thread_block();
  auto group_dim = g.group_dim();
  auto group_idx = g.group_index();
  auto idx = group_dim.x * group_idx.x + g.thread_rank();
  if (idx < N) {
    auto input_len = input_lens[idx];
    if (input_len == 0) {
      return;
    }
    const uint8_t *input = (uint8_t *)d_input + input_partial_sum_lens[idx];
    const uint8_t *input_bytes = (const uint8_t *)input;
    blake3_chunk_state *this_chunk = &d_hash[idx].chunk;
    blake3_hasher *self = &d_hash[idx];

    // If we have some partial chunk bytes in the internal chunk_state, we need
    // to finish that chunk first.
    size_t current_hash_len = chunk_state_len(g, d_hash);
    if (current_hash_len > 0) {
      size_t take = BLAKE3_CHUNK_LEN - current_hash_len;
      if (take > input_len) {
        take = input_len;
      }
      chunk_state_update(this_chunk, input_bytes, take);
      input_bytes += take;
      input_len -= take;
      // If we've filled the current chunk and there's more coming, finalize
      // this chunk and proceed. In this case we know it's not the root.
      if (input_len > 0) {
        output_t output = chunk_state_output(this_chunk);
        uint8_t chunk_cv[32];
        output_chaining_value(&output, chunk_cv);
        hasher_push_cv(self, chunk_cv, self->chunk.chunk_counter);
        chunk_state_reset(&self->chunk, self->key,
                          self->chunk.chunk_counter + 1);
      } else {
        return;
      }
    }

    // Now the chunk_state is clear, and we have more input. If there's more
    // than a single chunk (so, definitely not the root chunk), hash the largest
    // whole subtree we can, with the full benefits of SIMD (and maybe in the
    // future, multi-threading) parallelism. Two restrictions:
    // - The subtree has to be a power-of-2 number of chunks. Only subtrees
    // along
    //   the right edge can be incomplete, and we don't know where the right
    //   edge is going to be until we get to finalize().
    // - The subtree must evenly divide the total number of chunks up until this
    //   point (if total is not 0). If the current incomplete subtree is only
    //   waiting for 1 more chunk, we can't hash a subtree of 4 chunks. We have
    //   to complete the current subtree first.
    // Because we might need to break up the input to form powers of 2, or to
    // evenly divide what we already have, this part runs in a loop.
    while (input_len > BLAKE3_CHUNK_LEN) {
      size_t subtree_len = round_down_to_power_of_2(input_len);
      uint64_t count_so_far = this_chunk->chunk_counter * BLAKE3_CHUNK_LEN;
      // Shrink the subtree_len until it evenly divides the count so far. We
      // know that subtree_len itself is a power of 2, so we can use a
      // bitmasking trick instead of an actual remainder operation. (Note that
      // if the caller consistently passes power-of-2 inputs of the same size,
      // as is hopefully typical, this loop condition will always fail, and
      // subtree_len will always be the full length of the input.)
      //
      // An aside: We don't have to shrink subtree_len quite this much. For
      // example, if count_so_far is 1, we could pass 2 chunks to
      // compress_subtree_to_parent_node. Since we'll get 2 CVs back, we'll
      // still get the right answer in the end, and we might get to use 2-way
      // SIMD parallelism. The problem with this optimization, is that it gets
      // us stuck always hashing 2 chunks. The total number of chunks will
      // remain odd, and we'll never graduate to higher degrees of parallelism.
      // See https://github.com/BLAKE3-team/BLAKE3/issues/69.
      while ((((uint64_t)(subtree_len - 1)) & count_so_far) != 0) {
        subtree_len /= 2;
      }
      // The shrunken subtree_len might now be 1 chunk long. If so, hash that
      // one chunk by itself. Otherwise, compress the subtree into a pair of
      // CVs.
      uint64_t subtree_chunks = subtree_len / BLAKE3_CHUNK_LEN;
      if (subtree_len <= BLAKE3_CHUNK_LEN) {
        blake3_chunk_state chunk_state;
        chunk_state_init(&chunk_state, self->key, self->chunk.flags);
        chunk_state.chunk_counter = this_chunk->chunk_counter;
        chunk_state_update(&chunk_state, input_bytes, subtree_len);
        output_t output = chunk_state_output(&chunk_state);
        uint8_t cv[BLAKE3_OUT_LEN];
        output_chaining_value(&output, cv);
        hasher_push_cv(self, cv, chunk_state.chunk_counter);
      } else {
        // This is the high-performance happy path, though getting here depends
        // on the caller giving us a long enough input.
        uint8_t cv_pair[2 * BLAKE3_OUT_LEN];
        compress_subtree_to_parent_node(input_bytes, subtree_len, self->key,
                                        self->chunk.chunk_counter,
                                        self->chunk.flags, cv_pair);
        hasher_push_cv(self, cv_pair, self->chunk.chunk_counter);
        hasher_push_cv(self, &cv_pair[BLAKE3_OUT_LEN],
                       self->chunk.chunk_counter + (subtree_chunks / 2));
      }
      self->chunk.chunk_counter += subtree_chunks;
      input_bytes += subtree_len;
      input_len -= subtree_len;
    }

    // If there's any remaining input less than a full chunk, add it to the
    // chunk state. In that case, also do a final merge loop to make sure the
    // subtree stack doesn't contain any unmerged pairs. The remaining input
    // means we know these merges are non-root. This merge loop isn't strictly
    // necessary here, because hasher_push_chunk_cv already does its own merge
    // loop, but it simplifies blake3_hasher_finalize below.
    if (input_len > 0) {
      chunk_state_update(&self->chunk, input_bytes, input_len);
      hasher_merge_cv_stack(self, self->chunk.chunk_counter);
    }
  }
}

#ifdef __cplusplus
}
#endif
