#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <cuda.h>
#include <dlfcn.h>

#include <iostream>
#include <functional>

static void cuda_init() {
  static CUdevice device;
  static CUcontext context;
  static bool inited{false};

  if (!inited) {
    inited = true;
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);
  }
}

namespace py = pybind11;

// TODO f32
struct stencil_t {
  double *allocatedPtr;
  double *alignedPtr;
  int32_t offset;
  int32_t sizes[3];
  int32_t strides[3];
};

using stencil_fcn_t = void (*)(stencil_t *, stencil_t *);

static std::size_t compute_mem_size(py::buffer_info &info) {
  std::size_t size = info.itemsize;
  for (py::ssize_t i = 0; i < info.ndim; ++i) {
    size *= info.shape[i];
  }
  return size;
}

extern "C" {
  void mgpuMemAlloc(CUdeviceptr *ptr, uint64_t size);
  void mgpuMemFree(CUdeviceptr ptr);
}

static std::function<void(py::buffer, py::buffer)>
bind_stencil(std::string sym_name, std::string dl_name) {
  void *handle = dlopen(dl_name.c_str(), RTLD_LAZY | RTLD_NODELETE);
  if (char *err = dlerror()) {
    std::cerr << "dlopen(" << dl_name << ") error: " << err << std::endl;
    return nullptr;
  }
  std::string ciface_sym = "_mlir_ciface_" + sym_name;
  void *fcn_handle = dlsym(handle, ciface_sym.c_str());
  if (char *err = dlerror(); fcn_handle == nullptr) {
    std::cerr << "dlsym(" << ciface_sym << ") error: " << err << std::endl;
    return nullptr;
  }
  stencil_fcn_t stencil_fcn = reinterpret_cast<stencil_fcn_t>(fcn_handle);
  // TODO more than one input/output
  return [stencil_fcn](py::buffer input, py::buffer output) {
    py::buffer_info input_info = input.request();
    py::buffer_info output_info = output.request();

    if (input_info.ndim != 3) {
      throw std::runtime_error{"incompatible input shape: expected 3D array"};
    }
    if (output_info.ndim != 3) {
      throw std::runtime_error{"incompatible output shape: expected 3D array"};
    }

    if (input_info.format != py::format_descriptor<double>::format()) {
      throw std::runtime_error{"incompatible input format: expected f64"};
    }
    if (output_info.format != py::format_descriptor<double>::format()) {
      throw std::runtime_error{"incompatible output format: expected f64"};
    }

    std::size_t input_mem_size = compute_mem_size(input_info);
    std::size_t output_mem_size = compute_mem_size(output_info);

    CUdeviceptr input_mem_ptr{}, output_mem_ptr{};
    mgpuMemAlloc(&input_mem_ptr, input_mem_size);
    mgpuMemAlloc(&output_mem_ptr, output_mem_size);

    cuMemcpyHtoD(input_mem_ptr, input_info.ptr, input_mem_size);

    stencil_t input_stencil{
      (double *) input_mem_ptr, (double *) input_mem_ptr, 0,
      { input_info.shape[0], input_info.shape[1], input_info.shape[2] },
      { input_info.strides[0], input_info.strides[1], input_info.strides[2] }
    };
    stencil_t output_stencil{
      (double *) output_mem_ptr, (double *) output_mem_ptr, 0,
      { output_info.shape[0], output_info.shape[1], output_info.shape[2] },
      { output_info.strides[0], output_info.strides[1], output_info.strides[2] },
    };

    stencil_fcn(&input_stencil, &output_stencil);

    cuMemcpyDtoH(output_info.ptr, output_mem_ptr, output_mem_size);

    mgpuMemFree(input_mem_ptr);
    mgpuMemFree(input_mem_ptr);
  };
}

PYBIND11_MODULE(dl_stencil, m) {
  m.doc() = "Stencil Dynamic Library Binding";

  m.def("cuda_init", &cuda_init);
  m.def("bind_stencil", &bind_stencil);
}
