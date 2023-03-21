/* #include "blake3.h" */
#include <iostream>
#include <memory.h>
extern "C" void to_big_kernel();
extern "C" void pre_allocate(uint8_t device_id);
extern "C" void post_free(uint8_t device_id);
extern "C" void special_cuda_target(uint8_t *header, size_t start, size_t end,
                                    size_t stride, uint8_t target[32],
                                    uint64_t *host_randoms, uint32_t *found,
                                    uint8_t device_id);
extern "C" void getDeviceNum(int32_t *num);
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
  for (auto i = 8; i < 180; i += 4) {
    input[i] = 0xe7;
    input[i+1] = 0xf3;
    input[i+2] = 0x1b;
    input[i+3] = 0x08;
  }
  uint64_t host_randoms;
  uint32_t found = 0;

  special_cuda_target(input, 0x0102030405060708, 0x0f02030406060708, 1, target,
                      &host_randoms, &found, 0);
  printf("found:%d\n", found);

  post_free(0);
  delete[] input;
  delete[] target;
}
