#include "Type.h"
#include "Support.h"

#include <boost/python.hpp>

namespace mlir {
namespace py {

void exposeType() {
  using namespace boost::python;
  class_<Type>("Type", init<>())
      .def(self_ns::repr(self_ns::self))
      .def(self == self)
      .def(self != self)
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
