/* #include "blake3.h" */
#include <iostream>
#include <memory.h>
extern "C" void to_big_kernel();
extern "C" void pre_allocate();
extern "C" void post_free();
extern "C" void special_cuda_target(uint8_t *header, size_t start, size_t end,
                                    size_t stride, uint8_t target[32],
                                    uint64_t *host_randoms, uint32_t *host_len);
int main() {
  /* blake3_hasher hasher; */
  /* blake3_hasher_init(&hasher); */

  pre_allocate();
  /* to_big_kernel(); */
  uint8_t *input = new uint8_t[180];
  uint8_t *target = new uint8_t[32];
  for (auto i = 8; i < 180; i++) {
    input[i] = 0xe7;
  }
  uint64_t host_randoms[10];
  uint32_t host_len;

  special_cuda_target(input, 0x0102030405060708, 0x0f02030406060708, 1, target,
                      host_randoms, &host_len);

  post_free();
  delete[] input;
  delete[] target;
}
