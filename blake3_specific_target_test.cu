// includes, system
#include <cstdint>
#include <math.h>
#include <stdint.h>
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
#define PARALLEL_DEGREE 1024000

// internal flags
enum blake3_flags {
  CHUNK_START = 1 << 0,
  CHUNK_END = 1 << 1,
  PARENT = 1 << 2,
  ROOT = 1 << 3,
};

static uint8_t *pined_inp[8], *pined_target[8];
static uint32_t *pined_out[8];
static uint64_t *pined_randoms[8];
static bool *pined_found[8];
static cudaEvent_t event_start[8], event_stop[8];

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
__device__ __inline__ uint32_t to_bigend(uint32_t x) {
  return __byte_perm(x, x, 0x0123);
}

#define UPDATE_WITH_CV                                                         \
  do {                                                                         \
    S0 = 0x6A09E667UL;                                                         \
    S1 = 0xBB67AE85UL;                                                         \
    S2 = 0x3C6EF372UL;                                                         \
    S3 = 0xA54FF53AUL;                                                         \
    S4 = 0x510E527FUL;                                                         \
    S5 = 0x9B05688CUL;                                                         \
    S6 = 0x1F83D9ABUL;                                                         \
    S7 = 0x5BE0CD19UL;                                                         \
  } while (0);

#define UPDATE_WITH_CACHE                                                      \
  do {                                                                         \
    S0 = S0 ^ S8;                                                              \
    S1 = S1 ^ S9;                                                              \
    S2 = S2 ^ SA;                                                              \
    S3 = S3 ^ SB;                                                              \
    S4 = S4 ^ SC;                                                              \
    S5 = S5 ^ SD;                                                              \
    S6 = S6 ^ SE;                                                              \
    S7 = S7 ^ SF;                                                              \
  } while (0);

#define INIT(buf_len, flag)                                                    \
  do {                                                                         \
    S8 = 0x6A09E667UL;                                                         \
    S9 = 0xBB67AE85UL;                                                         \
    SA = 0x3C6EF372UL;                                                         \
    SB = 0xA54FF53AUL;                                                         \
    SC = 0;                                                                    \
    SD = 0;                                                                    \
    SE = (uint32_t)buf_len;                                                    \
    SF = (uint32_t)flag;                                                       \
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

#define UPDATE                                                                 \
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
    G(S3, S4, S9, SE, M[14], M[15]);                                           \
    G(S0, S4, S8, SC, M[2], M[6]);                                             \
    G(S1, S5, S9, SD, M[3], M[10]);                                            \
    G(S2, S6, SA, SE, M[7], M[0]);                                             \
    G(S3, S7, SB, SF, M[4], M[13]);                                            \
    G(S0, S5, SA, SF, M[1], M[11]);                                            \
    G(S1, S6, SB, SC, M[12], M[5]);                                            \
    G(S2, S7, S8, SD, M[9], M[14]);                                            \
    G(S3, S4, S9, SE, M[15], M[8]);                                            \
    G(S0, S4, S8, SC, M[3], M[4]);                                             \
    G(S1, S5, S9, SD, M[10], M[12]);                                           \
    G(S2, S6, SA, SE, M[13], M[2]);                                            \
    G(S3, S7, SB, SF, M[7], M[14]);                                            \
    G(S0, S5, SA, SF, M[6], M[5]);                                             \
    G(S1, S6, SB, SC, M[9], M[0]);                                             \
    G(S2, S7, S8, SD, M[11], M[15]);                                           \
    G(S3, S4, S9, SE, M[8], M[1]);                                             \
    G(S0, S4, S8, SC, M[10], M[7]);                                            \
    G(S1, S5, S9, SD, M[12], M[9]);                                            \
    G(S2, S6, SA, SE, M[14], M[3]);                                            \
    G(S3, S7, SB, SF, M[13], M[15]);                                           \
    G(S0, S5, SA, SF, M[4], M[0]);                                             \
    G(S1, S6, SB, SC, M[11], M[2]);                                            \
    G(S2, S7, S8, SD, M[5], M[8]);                                             \
    G(S3, S4, S9, SE, M[1], M[6]);                                             \
    G(S0, S4, S8, SC, M[12], M[13]);                                           \
    G(S1, S5, S9, SD, M[9], M[11]);                                            \
    G(S2, S6, SA, SE, M[15], M[10]);                                           \
    G(S3, S7, SB, SF, M[14], M[8]);                                            \
    G(S0, S5, SA, SF, M[7], M[2]);                                             \
    G(S1, S6, SB, SC, M[5], M[3]);                                             \
    G(S2, S7, S8, SD, M[0], M[1]);                                             \
    G(S3, S4, S9, SE, M[6], M[4]);                                             \
    G(S0, S4, S8, SC, M[9], M[14]);                                            \
    G(S1, S5, S9, SD, M[11], M[5]);                                            \
    G(S2, S6, SA, SE, M[8], M[12]);                                            \
    G(S3, S7, SB, SF, M[15], M[1]);                                            \
    G(S0, S5, SA, SF, M[13], M[3]);                                            \
    G(S1, S6, SB, SC, M[0], M[10]);                                            \
    G(S2, S7, S8, SD, M[2], M[6]);                                             \
    G(S3, S4, S9, SE, M[4], M[7]);                                             \
    G(S0, S4, S8, SC, M[11], M[15]);                                           \
    G(S1, S5, S9, SD, M[5], M[0]);                                             \
    G(S2, S6, SA, SE, M[1], M[9]);                                             \
    G(S3, S7, SB, SF, M[8], M[6]);                                             \
    G(S0, S5, SA, SF, M[14], M[10]);                                           \
    G(S1, S6, SB, SC, M[2], M[12]);                                            \
    G(S2, S7, S8, SD, M[3], M[4]);                                             \
    G(S3, S4, S9, SE, M[7], M[13]);                                            \
  } while (0);

#define CHECK_TARGET(N, P)                                                     \
  do {                                                                         \
    if ((((S##N ^ S##P) >> 0) & 0xff) > (*(d_target + N * 4 + 0)))             \
      return;                                                                  \
    if ((((S##N ^ S##P) >> 8) & 0xff) > (*(d_target + N * 4 + 1)))             \
      return;                                                                  \
    if ((((S##N ^ S##P) >> 16) & 0xff) > (*(d_target + N * 4 + 2)))            \
      return;                                                                  \
    if ((((S##N ^ S##P) >> 24) & 0xff) > (*(d_target + N * 4 + 3)))            \
      return;                                                                  \
  } while (0);

__global__ void special_launch(uint8_t *d_header, uint64_t start, uint64_t end,
                               size_t stride, uint8_t *d_target, uint32_t *out,
                               uint64_t *random_idx, bool *found) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t random_i = start + idx * stride; // parallel random message with i

  if (random_i < end) {
    // init chunk state
    // buf_len = 0, blocks_compressed = 0, flag = 0;
    uint32_t CV[8];
    uint32_t M[16] = {0}; // message blocks
    uint32_t S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, SA, SB, SC, SD, SE,
        SF; // the state var

    // process first block with 64B with 180 - 64 remain
    uint32_t h_random_i = random_i >> 32, low_random_i = (uint32_t)(random_i);
    M[0] = __byte_perm(h_random_i, h_random_i, 0x0123);
    M[1] = __byte_perm(low_random_i, low_random_i, 0x0123);
    for (auto i = 0; i < 14; i++) {
      M[i + 2] = *((uint32_t *)d_header + i);
    }
    /* printf("message: "); */
    /* for (auto i = 0; i < 64; i++) { */
    /*   printf("%02x", ((uint8_t *)M)[i]); */
    /* } */
    /* printf("\n"); */
    d_header += 56;

    // init states
    UPDATE_WITH_CV;
    INIT(BLAKE3_BLOCK_LEN, CHUNK_START);
    // round 0 - 6
    ROUND;
    // update chain value in place
    /* UPDAET; */

    /* printf("%d:%d,%d:%d,%d:%d,%d:%d,%d:%d,%d:%d,%d:%d,%d:%d,%d:%d,%d:%d,%d:%d,%"
     */
    /*        "d:%d,%d:%d,%d:%d,%d:%d,%d:%d,\n", */
    /*        0, S0, 1, S1, 2, S2, 3, S3, 4, S4, 5, S5, 6, S6, 7, S7, 8, S8, 9,
     * S9, */
    /*        10, SA, 11, SB, 12, SC, 13, SD, 14, SE, 15, SF); */

    // blocks_compressed = 1 remain 116
#pragma unroll
    for (auto i = 0; i < 16; i++) {
      M[i] = *((uint32_t *)d_header + i);
    }
    d_header += BLAKE3_BLOCK_LEN;

    // init states
    UPDATE_WITH_CACHE
    INIT(BLAKE3_BLOCK_LEN, 0);
    // round 0 - 6
    ROUND;
    // update chain value in place
    /* UPDATE; */
    // blocks_compressed = 2 remain 52 do final

#pragma unroll
    for (auto i = 0; i < 13; i++) {
      M[i] = *((uint32_t *)d_header + i);
    }

#pragma unroll
    for (auto i = 13; i < 16; i++) {
      M[i] = 0;
    }
    d_header += 52; // remain 0

    // init states
    UPDATE_WITH_CACHE
    INIT(52, CHUNK_END | ROOT);
    // round 0 - 6
    ROUND;
    UPDATE;
    // done output will be chain value

    // for debug
    uint32_t *self_out = out + idx * 8;
    self_out[0] = S0 ^ S8;
    self_out[1] = S1 ^ S9;
    self_out[2] = S2 ^ SA;
    self_out[3] = S3 ^ SB;
    self_out[4] = S4 ^ SC;
    self_out[5] = S5 ^ SD;
    self_out[6] = S6 ^ SE;
    self_out[7] = S7 ^ SF;

#pragma unroll
    for (auto i = 0; i < 32; i++) {
      if (((uint8_t *)&CV)[i] > ((uint8_t *)&d_target)[i])
        return;
      if (((uint8_t *)&CV)[i] < d_target[i]) {
        *found = true;
        *random_idx = (uint64_t)atomicMin((unsigned long long int *)random_idx,
                                          (unsigned long long int)random_i);
        return;
      }
    }

    // match i
    *found = true;
    // may be fault on mac,x32 system
    *random_idx = (uint64_t)atomicMin((unsigned long long int *)random_idx,
                                      (unsigned long long int)random_i);
  }
}

extern "C" void special_cuda_target(const uint8_t *header, uint64_t start,
                                    uint64_t end, size_t stride,
                                    const uint8_t target[32],
                                    uint64_t *host_randoms, bool *found,
                                    uint8_t device_id) {
  cudaProfilerStart();
  size_t num = (end - start) / stride;
  dim3 block;
  dim3 grid;
  if (num > 1024) {
    block = dim3(1024, 1, 1);
  } else {
    block = dim3(num, 1, 1);
  }
  grid = dim3(ceil(num * 1.0 / 1024), 1, 1);
  cudaEventRecord(event_start[device_id], 0);
  cudaMemcpyAsync(pined_inp[device_id], header + 8, INPUT_LEN - 8,
                  cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(pined_target[device_id], target, BLAKE3_OUT_LEN,
                  cudaMemcpyHostToDevice);
  cudaMemsetAsync(pined_found[device_id], 0, sizeof(bool));
  special_launch<<<1, 1024>>>(pined_inp[device_id], start, end, stride,
                              pined_target[device_id], pined_out[device_id],
                              pined_randoms[device_id], pined_found[device_id]);
  checkCudaErrors(cudaGetLastError());
  cudaMemcpyAsync(found, pined_found[device_id], sizeof(bool),
                  cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(host_randoms, pined_randoms[device_id], sizeof(uint64_t),
                  cudaMemcpyDeviceToHost);
  uint8_t *host_out = (uint8_t *)malloc(32 * 1024);
  cudaMemcpyAsync(host_out, pined_out[device_id], 32 * 1024,
                  cudaMemcpyDeviceToHost);
  cudaEventRecord(event_stop[device_id], 0);
  cudaProfilerStop();

  for (auto i = 0; i < 1; i++) {
    for (size_t i = 0; i < BLAKE3_OUT_LEN; i++) {
      printf("%02x", host_out[i]);
    }
    printf("\n");
    host_out += 32;
  }
}

extern "C" void pre_allocate(uint8_t device_id) {
  printf("allocate device %d\n", device_id);
  cudaSetDevice(device_id);
  cudaEventCreate(&event_start[device_id]);
  cudaEventCreate(&event_stop[device_id]);
  cudaMalloc((void **)&pined_inp[device_id], INPUT_LEN);
  cudaMalloc((void **)&pined_out[device_id], 32 * PARALLEL_DEGREE);
  cudaMalloc((void **)&pined_target[device_id], 32);
  cudaMalloc(&pined_found[device_id], sizeof(bool));
  cudaMalloc(&pined_randoms[device_id], sizeof(uint64_t) * 2);
}

extern "C" void post_free(uint8_t device_id) {
  cudaEventDestroy(event_start[device_id]);
  cudaEventDestroy(event_stop[device_id]);
  cudaFree(pined_inp[device_id]);
  cudaFree(pined_target[device_id]);
}

extern "C" void getDeviceNum(int32_t *nums) {
  cudaGetDeviceCount(nums);
  printf("detect %d nums gpu\n", *nums);
}

#ifdef __cplusplus
}
#endif
