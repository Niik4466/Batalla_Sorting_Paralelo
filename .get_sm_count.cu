#include <cuda_runtime.h>
#include <iostream>

int main(){
  int device, sm_count;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  printf("%d", sm_count);
  return 0;
}
