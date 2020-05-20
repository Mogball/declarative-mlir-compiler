#include "Type.h"
#include "Support.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

using namespace pybind11;

namespace mlir {
namespace py {

void exposeType(module &m) {
  class_<Type>(m, "Type")
      .def(init<>())
      .def(init<const Type &>())
      .def(self == self)
      .def(self != self)
      .def("__repr__", StringPrinter<Type>{})
      .def("__bool__", &Type::operator bool)
      .def("__invert__", &Type::operator!)
      .def("isIndex", &Type::isIndex)
      .def("isBF16", &Type::isBF16)
      .def("isF16", &Type::isF16)
      .def("isF32", &Type::isF32)
      .def("isF64", &Type::isF64)
      .def("isInteger", &Type::isInteger);
}

} // end namespace py
} // end namespace mlir