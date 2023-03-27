#include "blake3.h"
#include <iostream>
#include <memory.h>
extern "C" void to_big_kernel();
extern "C" void pre_allocate(uint8_t device_id);
extern "C" void special_cuda_target(uint8_t *header, size_t start, size_t end,
                                    uint64_t stride, uint8_t target[32],
                                    uint64_t *host_randoms, uint32_t *found,
                                    uint8_t device_id);
extern "C" void getDeviceNum(int32_t *num);

void cpu_blake3(uint8_t input[180], uint8_t target[32], uint64_t start,
                uint64_t end, size_t stride) {

  uint8_t *out = (uint8_t *)malloc(32);
  uint8_t *out_process = out;
  auto found = false;
  for (auto j = start; j < end; j += stride) {
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
    else{
      printf("cpu blake not found, random: %zu\n", random);
        for (auto i = 0; i < 32; i++) {
        printf("%02x", out_process[i]);
      }
      printf("\n");

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
     0, 0, 0, 0, 0, 0, 0, 1, 35, 124, 0, 0, 0, 0, 0, 0, 0, 0, 245, 37, 155, 93, 211, 128, 198, 33, 226, 112, 102, 65, 185, 146, 214, 89, 33, 202, 216, 155, 71, 32, 53, 134, 112, 23, 73, 51, 26, 62, 243, 145, 213, 147, 17, 243, 126, 237, 229, 83, 202, 224, 140, 205, 202, 85, 203, 139, 178, 170, 151, 57, 200, 49, 118, 92, 30, 50, 13, 77, 9, 90, 3, 74, 25, 188, 49, 32, 43, 46, 215, 23, 91, 154, 58, 7, 234, 171, 80, 15, 120, 245, 62, 68, 66, 223, 161, 149, 217, 154, 0, 0, 0, 0, 0, 26, 3, 223, 206, 242, 126, 154, 23, 43, 249, 158, 59, 202, 111, 141, 150, 225, 195, 160, 120, 16, 180, 214, 226, 193, 23, 41, 223, 136, 84, 30, 135, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  uint8_t target[32] = {63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};
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
  uint64_t start = 1;
  uint64_t end = 3;
  size_t stride = 1;
  cpu_blake3(input, target, start, end, stride);
  special_cuda_target(input, start, end, stride, target, &host_randoms, &found,
                      0);
  printf("gpu found:%d, random: %zu\n", found, host_randoms);
}
