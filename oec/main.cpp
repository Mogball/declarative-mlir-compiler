#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cuda.h>

extern "C" {
  typedef struct {
    double *allocatedPtr;
    double *alignedPtr;
    int32_t offset;
    int32_t sizes[3];
    int32_t strides[3];
  } MemRefType3D;

  void mgpuMemAlloc(CUdeviceptr *ptr, uint64_t size);
  void mgpuMemFree(CUdeviceptr ptr);

  void _mlir_ciface_laplace(MemRefType3D *input, MemRefType3D *output);
  void _mlir_ciface_fill(MemRefType3D *inout);
  double _mlir_ciface_get(MemRefType3D *input, int32_t i, int32_t j, int32_t k);
}

int main() {
  cuInit(0);
  CUdevice device;
  cuDeviceGet(&device, 0);
  CUcontext context;
  cuCtxCreate(&context, 0, device);

  constexpr int32_t dim = 72;
  constexpr auto mem_size = dim * dim * dim * sizeof(double);
  constexpr auto mem_align = sizeof(double);

  CUdeviceptr input_mem{}, output_mem{};
  std::size_t input_space = mem_size + mem_align;
  std::size_t output_space = input_space;
  mgpuMemAlloc(&input_mem, input_space);
  mgpuMemAlloc(&output_mem, output_space);

  auto *input_ptr = (void *) input_mem;
  if (!std::align(mem_align, mem_size, input_ptr, input_space)) {
    std::cout << "Failed to align input memory" << std::endl;
    return -1;
  }

  auto *output_ptr = (void *) output_mem;
  if (!std::align(mem_align, mem_size, output_ptr, output_space)) {
    std::cout << "Failed to align output memory" << std::endl;
    return -1;
  }

  MemRefType3D input{ (double *) input_mem, (double *) input_ptr, 0,
                      { dim, dim, dim },
                      { dim * dim * sizeof(double), dim * sizeof(double),
                        sizeof(double) }};

  MemRefType3D output{ (double *) output_mem, (double *) output_ptr, 0,
                       { dim, dim, dim },
                       { dim * dim * sizeof(double), dim * sizeof(double),
                         sizeof(double) }};

  _mlir_ciface_fill(&input);
  _mlir_ciface_laplace(&input, &output);

  mgpuMemFree(input_mem);
  mgpuMemFree(output_mem);
  return 0;
}
