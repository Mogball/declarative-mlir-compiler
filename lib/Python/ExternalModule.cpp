#include "dmc/Python/PyMLIR.h"

PYBIND11_MODULE(mlir, m) {
  mlir::py::getModule(m);
}
