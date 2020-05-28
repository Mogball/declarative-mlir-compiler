#include "Expose.h"

PYBIND11_MODULE(mlir, m) {
  mlir::py::expose(m);
}
