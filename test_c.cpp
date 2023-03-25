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
      for(auto i = 0;i < 32 ;i++){
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
  uint8_t *input = new uint8_t[180];
  uint8_t *target = new uint8_t[32];
  for (auto i = 0; i < 32; i++) {
    target[i] = 0x01;
  }
  for (auto i = 8; i < 180; i += 4) {
    input[i] = 0xe7;
    input[i + 1] = 0xf3;
    input[i + 2] = 0x1b;
    input[i + 3] = 0x08;
  }
  uint64_t host_randoms;
  uint32_t found = 0;
  uint64_t start = 100;
  uint64_t end = 1024*10240;
  size_t stride = 1;
  cpu_blake3(input, target, start, end, stride);
  special_cuda_target(input, start, end, stride, target, &host_randoms, &found,
                      0);
  printf("gpu found:%d, random: %zu\n", found, host_randoms);

  delete[] input;
  delete[] target;
}
