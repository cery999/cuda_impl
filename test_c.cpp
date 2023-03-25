#include "blake3.h"
#include <iostream>
#include <memory.h>
extern "C" void to_big_kernel();
extern "C" void pre_allocate(uint8_t device_id);
extern "C" void special_cuda_target(uint8_t *header, size_t start, size_t end,
                                    size_t stride, uint8_t target[32],
                                    uint64_t *host_randoms, uint32_t *found,
                                    uint8_t device_id);
extern "C" void getDeviceNum(int32_t *num);

void cpu_blake3(uint8_t input[180], uint8_t target[32], uint64_t start,
                uint64_t end, size_t stride) {

  uint8_t *out = (uint8_t *)malloc(32);
  uint8_t *out_process = out;
  auto found = false;
  for (auto j = start; j <= end; j += stride) {
    // Initialize the hasher.
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    uint64_t x = htobe64(j);
    uint8_t *ptr = (uint8_t *)&x;
    for (int i = 0; i < 8; i++) {
      input[i] = ptr[i];
    }
    blake3_hasher_update(&hasher, input, 180);

    // Finalize the hash. BLAKE3_OUT_LEN is the default output length, 32 bytes.
    uint8_t output[BLAKE3_OUT_LEN];
    blake3_hasher_finalize(&hasher, output, BLAKE3_OUT_LEN);

    // Print the hash as hexadecimal.
    for (size_t i = 0; i < BLAKE3_OUT_LEN; i++) {
      out_process[i] = output[i];
    }
    auto random = j;
    auto is_break = false;
    found = false;
    for (auto i = 0; i < BLAKE3_OUT_LEN; i++) {
      if (out_process[i] < target[i]) {
        is_break = true;
        found = true;
        break;
      }
      if (out_process[i] > target[i]) {
        found = false;
        is_break = true;
        break;
      }
    }
    if (!is_break) {
      found = true;
    }
    if (found) {
      printf("cpu blake found: 1 random: %zu\n", random);
      for (auto i = 0; i < 32; i++) {
        printf("%02x", out_process[i]);
      }
      printf("\n");
      break;
    }
  }
  if (!found)
    printf("cpu not found \n");
}

int main() {
  /* blake3_hasher hasher; */
  /* blake3_hasher_init(&hasher); */
  int32_t num = 0;
  getDeviceNum(&num);
  printf("start task with %d gpus\n", num);
  pre_allocate(0);
  /* to_big_kernel(); */
  uint8_t input[180] = {
      0,   0,   0,   0,   0,   0,   0,   0,   65,  118, 0,   0,   0,   0,   0,
      0,   0,   22,  20,  132, 172, 9,   113, 114, 13,  61,  65,  36,  33,  163,
      233, 83,  72,  65,  202, 201, 217, 113, 109, 84,  252, 244, 193, 44,  235,
      132, 220, 158, 203, 41,  151, 77,  104, 223, 89,  6,   69,  27,  71,  73,
      112, 102, 188, 214, 24,  229, 214, 112, 134, 61,  187, 246, 222, 32,  148,
      46,  113, 198, 222, 50,  21,  2,   3,   218, 222, 213, 88,  236, 82,  51,
      194, 7,   96,  2,   184, 197, 34,  8,   229, 238, 225, 220, 143, 250, 90,
      152, 87,  107, 0,   0,   0,   0,   0,   25,  150, 183, 59,  167, 24,  209,
      179, 204, 43,  160, 141, 212, 249, 114, 27,  37,  189, 94,  236, 150, 161,
      237, 119, 144, 55,  95,  157, 148, 240, 24,  135, 1,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
  uint8_t target[32] = {0,   0,   0,   0,   0,   25,  150, 183, 59,  167, 24,
                        209, 179, 204, 43,  160, 141, 212, 249, 114, 27,  37,
                        189, 94,  236, 150, 161, 237, 119, 144, 55,  95};
  /* for (auto i = 0; i < 32; i++) { */
  /*   target[i] = 0x01; */
  /* } */
  /* for (auto i = 8; i < 180; i += 4) { */
  /*   input[i] = 0xe7; */
  /*   input[i + 1] = 0xf3; */
  /*   input[i + 2] = 0x1b; */
  /*   input[i + 3] = 0x08; */
  /* } */
  uint64_t host_randoms;
  uint32_t found = 0;
  uint64_t start = 3208642560;
  uint64_t end = 3219128320;
  size_t stride = 1;
  cpu_blake3(input, target, start, end, stride);
  special_cuda_target(input, start, end, stride, target, &host_randoms, &found,
                      0);
  printf("gpu found:%d, random: %zu\n", found, host_randoms);
}
