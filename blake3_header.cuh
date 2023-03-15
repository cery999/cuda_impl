#pragma once
// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// internal flags
enum blake3_flags {
  CHUNK_START = 1 << 0,
  CHUNK_END = 1 << 1,
  PARENT = 1 << 2,
  ROOT = 1 << 3,
};

// This struct is a private implementation detail. It has to be here because
// it's part of blake3_hasher below.
// 8*4+8+64+3 = 107
typedef struct __align__(4) {
  uint32_t cv[8];
  uint8_t buf[BLAKE3_BLOCK_LEN];
  uint64_t chunk_counter;
  uint8_t buf_len;
  uint8_t blocks_compressed;
  uint8_t flags;
}
blake3_chunk_state;

// 4*8 + 107 + 1+ 55*32 = 1900 B
typedef struct __align__(4) {
  uint32_t key[8];
  blake3_chunk_state chunk;
  // The stack size is MAX_DEPTH + 1 because we do lazy merging. For example,
  // with 7 chunks, we have 3 entries in the stack. Adding an 8th chunk
  // requires a 4th entry, rather than merging everything down to 1, because we
  // don't know whether more input is coming. This is different from how the
  // reference implementation does things.
  uint8_t cv_stack[(BLAKE3_MAX_DEPTH + 1) * BLAKE3_OUT_LEN];
  uint8_t cv_stack_len;
}
blake3_hasher;

typedef struct __align__(4) {
  uint32_t input_cv[8];
  uint64_t counter;
  uint8_t block[BLAKE3_BLOCK_LEN];
  uint8_t block_len;
  uint8_t flags;
}
output_t;

__constant__ uint32_t __align__(4) IV[8]{
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL};

__constant__ uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

static blake3_hasher *pined_d_hasher;

extern "C" void blake3_hasher_init_cuda(blake3_hasher *data, size_t N);
extern "C" void blake3_hasher_reset_cuda(blake3_hasher *data, size_t N);
extern "C" void blake3_hasher_update_cuda(blake3_hasher *data,
                                          const uint8_t *const *inputs,
                                          size_t *input_lens, size_t N);
extern "C" void pre_allocate();
extern "C" void post_free();

#ifdef __cplusplus
}
#endif
