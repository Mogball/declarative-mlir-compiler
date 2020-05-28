#pragma once

#include <pybind11/pybind11.h>

namespace mlir {
class MLIRContext;
namespace py {
void getModule(pybind11::module &m);
void setMLIRContext(MLIRContext *ctx);
MLIRContext *getMLIRContext();
} // end namespace py
} // end namespace mlir
