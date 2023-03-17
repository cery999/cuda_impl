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

// includes, project
#include <helper_cuda.h> // helper functions for CUDA error checking and initialization
#include <helper_functions.h> // helper utility functions

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
// internal flags
enum blake3_flags {
  CHUNK_START = 1 << 0,
  CHUNK_END = 1 << 1,
  PARENT = 1 << 2,
  ROOT = 1 << 3,
};

static void *pined_inp, *pined_target;

__device__ __inline__ uint64_t to_big_end(uint64_t x) {
  // 1 2 3 4 5 6 7 8  -> 8 7 6 5 4 3 2 1
  return ((uint64_t)__byte_perm((uint32_t)(x), (uint32_t)(x >> 32), 0x0123)
          << 32) |
         __byte_perm((uint32_t)(x), (uint32_t)(x >> 32), 0x4567);
}

__global__ void to_big_end_launch(uint64_t *a, uint64_t *out) {
  *out = to_big_end(*a);
}

extern "C" void to_big_kernel() {
  void *a, *out;
  cudaMalloc(&a, sizeof(uint64_t));
  cudaMalloc(&out, sizeof(uint64_t));
  uint64_t l = 0x0102030405060708;
  uint8_t *ptr_l = (uint8_t *)&l;
  for (int i = 0; i < 8; i++) {
    printf("%02x", ptr_l[i]);
  }
  printf("\n");
  cudaMemcpy(a, &l, sizeof(uint64_t), cudaMemcpyHostToDevice);
  to_big_end_launch<<<1, 1>>>((uint64_t *)a, (uint64_t *)out);
  uint64_t oo;
  cudaMemcpy(&oo, out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  uint8_t *ptr = (uint8_t *)&oo;
  for (int i = 0; i < 8; i++) {
    printf("%02x", ptr[i]);
  }
  printf("\n");
  cudaFree(a);
  cudaFree(out);
}
__device__ uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
}

#define INIT(buf_len, flag)                                                    \
  do {                                                                         \
    S0 = CV[0];                                                                \
    S1 = CV[1];                                                                \
    S2 = CV[2];                                                                \
    S3 = CV[3];                                                                \
    S4 = CV[4];                                                                \
    S5 = CV[5];                                                                \
    S6 = CV[6];                                                                \
    S7 = CV[7];                                                                \
    S8 = 0x6A09E667UL;                                                         \
    S9 = 0xBB67AE85UL;                                                         \
    SA = 0x3C6EF372UL;                                                         \
    SB = 0xA54FF53AUL;                                                         \
    SC = 0;                                                                    \
    SD = 0;                                                                    \
    SE = buf_len;                                                              \
    SF = flag;                                                                 \
  } while (0);

#define G(a, b, c, d, x, y)                                                    \
  do {                                                                         \
    a = a + b + x;                                                             \
    d = rotr32(d ^ a, 16);                                                     \
    c = c + d;                                                                 \
    b = rotr32(b ^ c, 12);                                                     \
    a = a + b + y;                                                             \
    d = rotr32(d ^ a, 8);                                                      \
    c = c + d;                                                                 \
    b = rotr32(b ^ c, 7);                                                      \
  } while (0);

#define UPDAET                                                                 \
  do {                                                                         \
    CV[0] = S0 ^ S8;                                                           \
    CV[1] = S1 ^ S9;                                                           \
    CV[2] = S2 ^ SA;                                                           \
    CV[3] = S3 ^ SB;                                                           \
    CV[4] = S4 ^ SC;                                                           \
    CV[5] = S5 ^ SD;                                                           \
    CV[6] = S6 ^ SE;                                                           \
    CV[7] = S7 ^ SF;                                                           \
  } while (0);

#define ROUND                                                                  \
  do {                                                                         \
    G(S0, S4, S8, SC, M[0], M[1]);                                             \
    G(S1, S5, S9, SD, M[2], M[3]);                                             \
    G(S2, S6, SA, SE, M[4], M[5]);                                             \
    G(S3, S7, SB, SF, M[6], M[7]);                                             \
    G(S0, S5, SA, SF, M[8], M[9]);                                             \
    G(S1, S6, SB, SC, M[10], M[11]);                                           \
    G(S2, S7, S8, SD, M[12], M[13]);                                           \
    G(S3, S7, S9, SE, M[14], M[15]);                                           \
    G(S0, S4, S8, SC, M[2], M[6]);                                             \
    G(S1, S5, S9, SD, M[3], M[10]);                                            \
    G(S2, S6, SA, SE, M[7], M[0]);                                             \
    G(S3, S7, SB, SF, M[4], M[13]);                                            \
    G(S0, S5, SA, SF, M[1], M[11]);                                            \
    G(S1, S6, SB, SC, M[12], M[5]);                                            \
    G(S2, S7, S8, SD, M[9], M[14]);                                            \
    G(S3, S7, S9, SE, M[15], M[8]);                                            \
    G(S0, S4, S8, SC, M[3], M[4]);                                             \
    G(S1, S5, S9, SD, M[10], M[12]);                                           \
    G(S2, S6, SA, SE, M[13], M[2]);                                            \
    G(S3, S7, SB, SF, M[7], M[14]);                                            \
    G(S0, S5, SA, SF, M[6], M[5]);                                             \
    G(S1, S6, SB, SC, M[9], M[0]);                                             \
    G(S2, S7, S8, SD, M[11], M[15]);                                           \
    G(S3, S7, S9, SE, M[8], M[1]);                                             \
    G(S0, S4, S8, SC, M[10], M[7]);                                            \
    G(S1, S5, S9, SD, M[12], M[9]);                                            \
    G(S2, S6, SA, SE, M[14], M[3]);                                            \
    G(S3, S7, SB, SF, M[13], M[15]);                                           \
    G(S0, S5, SA, SF, M[4], M[0]);                                             \
    G(S1, S6, SB, SC, M[11], M[2]);                                            \
    G(S2, S7, S8, SD, M[5], M[8]);                                             \
    G(S3, S7, S9, SE, M[1], M[6]);                                             \
    G(S0, S4, S8, SC, M[12], M[13]);                                           \
    G(S1, S5, S9, SD, M[9], M[11]);                                            \
    G(S2, S6, SA, SE, M[15], M[10]);                                           \
    G(S3, S7, SB, SF, M[14], M[8]);                                            \
    G(S0, S5, SA, SF, M[7], M[2]);                                             \
    G(S1, S6, SB, SC, M[5], M[3]);                                             \
    G(S2, S7, S8, SD, M[0], M[1]);                                             \
    G(S3, S7, S9, SE, M[6], M[4]);                                             \
    G(S0, S4, S8, SC, M[9], M[14]);                                            \
    G(S1, S5, S9, SD, M[11], M[5]);                                            \
    G(S2, S6, SA, SE, M[8], M[12]);                                            \
    G(S3, S7, SB, SF, M[15], M[1]);                                            \
    G(S0, S5, SA, SF, M[13], M[3]);                                            \
    G(S1, S6, SB, SC, M[0], M[10]);                                            \
    G(S2, S7, S8, SD, M[2], M[6]);                                             \
    G(S3, S7, S9, SE, M[4], M[7]);                                             \
    G(S0, S4, S8, SC, M[11], M[15]);                                           \
    G(S1, S5, S9, SD, M[5], M[0]);                                             \
    G(S2, S6, SA, SE, M[1], M[9]);                                             \
    G(S3, S7, SB, SF, M[8], M[6]);                                             \
    G(S0, S5, SA, SF, M[14], M[10]);                                           \
    G(S1, S6, SB, SC, M[2], M[12]);                                            \
    G(S2, S7, S8, SD, M[3], M[4]);                                             \
    G(S3, S7, S9, SE, M[7], M[13]);                                            \
  } while (0);

__global__ void special_launch(uint8_t *d_header, size_t start, size_t end,
                               size_t stride, uint8_t *d_target) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  // init chunk state
  auto buf_len = 0, blocks_compressed = 0, flag = 0;
  uint32_t CV[8] = {0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL,
                    0xA54FF53AUL, 0x510E527FUL, 0x9B05688CUL,
                    0x1F83D9ABUL, 0x5BE0CD19UL}; // cv
  uint32_t M[16] = {0};                          // message blocks
  uint32_t S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, SA, SB, SC, SD, SE,
      SF; // the state var

  uint64_t random_i = start + idx * stride; // parallel random message with i

  // process first block with 64B with 180 - 64 remain
  uint32_t h_random_i = random_i >> 32, low_random_i = (uint32_t)(random_i);
  M[0] = __byte_perm(low_random_i, h_random_i, 0x0123);
  M[1] = __byte_perm(low_random_i, h_random_i, 0x4567);
  d_header += 8;
  memcpy(&M[2], d_header, 56); // copy 8 + 56 = 64B
  d_header += 56;

  // init states
  INIT(BLAKE3_BLOCK_LEN, CHUNK_START);
  // round 0 - 6
  ROUND;
  // update chain value in place
  UPDAET;

  // blocks_compressed = 1 remain 116
  memcpy(&M[0], d_header, BLAKE3_BLOCK_LEN); // copy 64B
  d_header += BLAKE3_BLOCK_LEN;

  // init states
  INIT(BLAKE3_BLOCK_LEN, 0);
  // round 0 - 6
  ROUND;
  // update chain value in place
  UPDAET;
  // blocks_compressed = 2 remain 52 do final
}

void special_cuda_target(uint8_t *header, size_t start, size_t end,
                         size_t stride, uint8_t target[32]) {
  checkCudaErrors(cudaProfilerStart());
  cudaMemcpyAsync(pined_inp, header, INPUT_LEN, cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(pined_target, target, BLAKE3_OUT_LEN, cudaMemcpyHostToDevice,
                  0);

  checkCudaErrors(cudaProfilerStop());
}

extern "C" void pre_allocate() {
  checkCudaErrors(cudaMalloc((void **)&pined_inp, INPUT_LEN));
}

extern "C" void post_free() { checkCudaErrors(cudaFree(pined_inp)); }

#ifdef __cplusplus
}
#endif
