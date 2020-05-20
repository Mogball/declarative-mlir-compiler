#pragma once

#include <pybind11/pybind11.h>

namespace mlir {
namespace py {

void exposeParser(pybind11::module &m);
void exposeModule(pybind11::module &m);
void exposeLocation(pybind11::module &m);
void exposeType(pybind11::module &m);

} // end namespace py
} // end namespace mlir
