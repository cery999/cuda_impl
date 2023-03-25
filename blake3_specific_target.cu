// includes, system
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA runtime
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
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
#define PARALLEL_DEGREE 10240

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

__device__ uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
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
    d = __byte_perm(d ^ a, d ^ a, 0x1032);                                     \
    c = c + d;                                                                 \
    b = rotr32(b ^ c, 12);                                                     \
    a = a + b + y;                                                             \
    d = __byte_perm(d ^ a, d ^ a, 0x0321);                                     \
    c = c + d;                                                                 \
    b = rotr32(b ^ c, 7);                                                      \
  } while (0);

#define UPDATE                                                                 \
  do {                                                                         \
    *reinterpret_cast<uint4 *>(&CV[0]) = make_uint4(S0, S1, S2, S3);           \
    *reinterpret_cast<uint4 *>(&CV[4]) = make_uint4(S4, S5, S6, S7);           \
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

__global__ void special_launch(uint8_t *d_header, uint64_t start, uint64_t end,
                               size_t stride, uint8_t *d_target, uint32_t *out,
                               uint64_t *block_random_idx, bool *block_found) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t random_i = start + idx * stride; // parallel random message with i
  auto found = false;
  auto cta = this_thread_block();
  auto tile = tiled_partition<32>(cta);
  auto grid = this_grid();

  __shared__ bool thread_tile_group_found[32 + 1];
  __shared__ uint64_t thread_tile_group_random[32 + 1];
  if (random_i < end) {
    // init chunk state
    // buf_len = 0, blocks_compressed = 0, flag = 0;
    uint32_t M[16] = {0}; // message blocks
    uint32_t S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, SA, SB, SC, SD, SE,
        SF; // the state var

    // process first block with 64B with 180 - 64 remain
    uint32_t h_random_i = random_i >> 32, low_random_i = (uint32_t)(random_i);
    M[0] = __byte_perm(h_random_i, h_random_i, 0x0123);
    M[1] = __byte_perm(low_random_i, low_random_i, 0x0123);
    for (auto i = 0; i < 3; i++) {
      *(reinterpret_cast<int4 *>(&M[i * 4 + 2])) =
          *(reinterpret_cast<int4 *>(&d_header[i * 16]));
    }
    *(reinterpret_cast<int2 *>(&M[14])) =
        *(reinterpret_cast<int2 *>(&d_header[48]));

    UPDATE_WITH_CV;
    INIT(BLAKE3_BLOCK_LEN, CHUNK_START);
    ROUND;

#pragma unroll
    for (auto i = 0; i < 4; i++) {
      *(reinterpret_cast<int4 *>(&M[i * 4])) =
          *(reinterpret_cast<int4 *>(&d_header[i * 16 + 56]));
    }

    UPDATE_WITH_CACHE;
    INIT(BLAKE3_BLOCK_LEN, 0);
    ROUND;

#pragma unroll
    for (auto i = 0; i < 3; i++) {
      *(reinterpret_cast<int4 *>(&M[i * 4])) =
          *(reinterpret_cast<int4 *>(&d_header[i * 16 + 56 + 64]));
    }
    *(reinterpret_cast<int2 *>(&M[13])) = make_int2(0, 0);
    M[15] = 0;

    // init states
    UPDATE_WITH_CACHE;
    INIT(52, CHUNK_END | ROOT);
    ROUND;
    /* UPDATE; */
    UPDATE_WITH_CACHE;

    uint32_t CV[8];
    *reinterpret_cast<uint4 *>(&CV[0]) = make_uint4(S0, S1, S2, S3);
    *reinterpret_cast<uint4 *>(&CV[4]) = make_uint4(S4, S5, S6, S7);

    auto is_break = false;
    for (auto i = 0; i < 32; i++) {
      if (((uint8_t *)CV)[i] < d_target[i]) {
        is_break = true;
        found = true;
        break;
      }
      if (((uint8_t *)CV)[i] > d_target[i]) {
        is_break = true;
        found = false;
        break;
      }
    }
    if (!is_break) {
      found = true; // equal
    }
  }

  bool warp_found = false;
  uint64_t warp_random_idx = found ? random_i : UINT64_MAX;
  warp_found = tile.any(found);
  warp_random_idx = reduce(tile, warp_random_idx, less<uint64_t>());

  if (tile.thread_rank() == 0) {
    thread_tile_group_found[tile.meta_group_rank()] = warp_found;
    thread_tile_group_random[tile.meta_group_rank()] = warp_random_idx;
  }
  sync(cta);

  if (tile.meta_group_rank() == 0) {
    bool warp_group_found = false;
    uint64_t warp_group_random = UINT64_MAX;
    warp_group_found = thread_tile_group_found[tile.thread_rank()];
    warp_group_random = thread_tile_group_random[tile.thread_rank()];

    warp_group_found = tile.any(warp_group_found);
    warp_group_random = reduce(tile, warp_group_random, less<uint64_t>());
    /* for (auto offset = 16; offset > 0; offset >>= 1) { */
    /*   warp_group_found |= */
    /*       __shfl_down_sync(0x0000ffff, warp_group_found, offset); */
    /*   warp_group_random = */
    /*       min(__shfl_down_sync(0x0000ffff, warp_group_random, offset), */
    /*           warp_group_random); */
    /* } */

    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
      block_found[grid.block_rank()] = warp_group_found;
      block_random_idx[grid.block_rank()] = warp_group_random;
    }
  }
}

__global__ void reduceGlobalBlocks(bool *global_found, uint64_t *global_random,
                                   uint64_t num) {
  volatile __shared__ bool shared_found[32];
  volatile __shared__ uint64_t shared_random[32];
  auto block = this_thread_block();

  unsigned int tid = threadIdx.x;
  unsigned int gridSize = block.size() * gridDim.x;
  unsigned int maskLength = (block.size() & 31); // 31 = warpSize-1
  maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
  const unsigned int mask = (0xffffffff) >> maskLength;

  bool found = false;
  uint64_t random = UINT64_MAX;
  if ((block.size() & (block.size() - 1)) == 0) {
    unsigned int i = blockIdx.x * block.size() * 2 + threadIdx.x;
    gridSize = gridSize << 1;
    while (i < num) {
      found |= global_found[i];
      random = min(global_random[i], random);

      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + block.size()) < num) {
        found |= global_found[i + block.size()];
        random = min(global_random[i + block.size()], random);
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * block.size() + threadIdx.x;
    while (i < num) {
      found |= global_found[i];
      random = min(random, global_random[i]);
      i += gridSize;
    }
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    auto warp_found = found;
    auto warp_random = random;
    warp_found |= __shfl_down_sync(mask, warp_found, offset);
    warp_random = min(__shfl_down_sync(mask, warp_random, offset), warp_random);
    if (threadIdx.x + offset < block.size()) {
      found = warp_found;
      random = warp_random;
    }
  }

  if ((tid % warpSize) == 0) {
    shared_found[tid / warpSize] = found;
    shared_random[tid / warpSize] = random;
  }
  __syncthreads();

  const unsigned int shmem_extent =
      block.size() / warpSize > 0 ? block.size() / warpSize : 1;
  const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    found = shared_found[tid];
    random = shared_random[tid];

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      auto warp_group_found = found;
      auto warp_group_random = random;
      warp_group_found |=
          __shfl_down_sync(ballot_result, warp_group_found, offset);
      warp_group_random =
          min(warp_group_random,
              __shfl_down_sync(ballot_result, warp_group_random, offset));
      if ((tid + offset) < shmem_extent) {
        found = warp_group_found;
        random = warp_group_random;
      }
    }
  }

  if (tid == 0) {
    global_found[blockIdx.x] = found;
    global_random[blockIdx.x] = random;
  }
}

extern "C" void special_cuda_target(const uint8_t *header, uint64_t start,
                                    uint64_t end, size_t stride,
                                    const uint8_t target[32],
                                    uint64_t *host_randoms, uint32_t *found,
                                    uint8_t device_id) {
  size_t num = ceil((end - start) * 1.0 / stride);
  dim3 block;
  dim3 grid;
  block = dim3(1024, 1, 1);
  grid = dim3(ceil(num * 1.0 / 1024), 1, 1);
  cudaEventRecord(event_start[device_id], 0);
  cudaMemcpyAsync(pined_inp[device_id], header + 8, INPUT_LEN - 8,
                  cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(pined_target[device_id], target, BLAKE3_OUT_LEN,
                  cudaMemcpyHostToDevice);
  special_launch<<<grid, block>>>(
      pined_inp[device_id], start, end, stride, pined_target[device_id],
      pined_out[device_id], pined_randoms[device_id], pined_found[device_id]);

  auto total_block_num = grid.x;
  if (total_block_num >= 1024) {
    block = dim3(1024, 1, 1);
  } else {
    block = dim3(total_block_num, 1, 1);
  }
  grid = dim3(ceil((total_block_num * 1.0) / 1024), 1, 1);
  if (block.x > 1) {
    reduceGlobalBlocks<<<grid, block>>>(pined_found[device_id],
                                        pined_randoms[device_id], grid.x);
  }
  bool pined_host_found;
  cudaMemcpyAsync(&pined_host_found, pined_found[device_id], sizeof(bool),
                  cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(host_randoms, pined_randoms[device_id], sizeof(uint64_t),
                  cudaMemcpyDeviceToHost);
  cudaEventRecord(event_stop[device_id], 0);
  cudaEventSynchronize(event_stop[device_id]);
  *found = (uint32_t)pined_host_found;
}

extern "C" void pre_allocate(uint8_t device_id) {
  printf("allocate device %d\n", device_id);
  cudaSetDevice(device_id);
  cudaEventCreate(&event_start[device_id]);
  cudaEventCreate(&event_stop[device_id]);
  cudaMalloc((void **)&pined_inp[device_id], INPUT_LEN);
  cudaMalloc((void **)&pined_target[device_id], 32);
  cudaMalloc(&pined_found[device_id], sizeof(bool) * PARALLEL_DEGREE);
  cudaMalloc(&pined_randoms[device_id], sizeof(uint64_t) * PARALLEL_DEGREE);
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
