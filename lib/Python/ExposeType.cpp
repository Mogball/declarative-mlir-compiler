#include "Type.h"
#include "Support.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

using namespace pybind11;
using namespace llvm;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "type");
}

void exposeType(module &m) {
  class_<Type>(m, "Type")
      .def(init<>())
      .def(init<const Type &>())
      .def(self == self)
      .def(self != self)
      .def("__repr__", StringPrinter<Type>{})
      .def("__bool__", &Type::operator bool)
      .def("__invert__", &Type::operator!)
      .def("__hash__", overload<hash_code(Type)>(&hash_value))
      .def("isIndex", nullcheck(&Type::isIndex))
      .def("isBF16", nullcheck(&Type::isBF16))
      .def("isF16", nullcheck(&Type::isF16))
      .def("isF32", nullcheck(&Type::isF32))
      .def("isF64", nullcheck(&Type::isF64))
      .def("isInteger", nullcheck(&Type::isInteger))
      .def("isSignlessInteger", nullcheck(
          overload<bool(Type::*)()>(&Type::isSignlessInteger)))
      .def("isSignlessInteger", nullcheck(
          overload<bool(Type::*)(unsigned)>(&Type::isSignlessInteger)))
      .def("isSignedInteger", nullcheck(
          overload<bool(Type::*)()>(&Type::isSignedInteger)))
      .def("isSignedInteger", nullcheck(
          overload<bool(Type::*)(unsigned)>(&Type::isSignedInteger)))
      .def("isUnsignedInteger", nullcheck(
          overload<bool(Type::*)()>(&Type::isSignedInteger)))
      .def("isUnsignedInteger", nullcheck(
          overload<bool(Type::*)(unsigned)>(&Type::isSignedInteger)))
      .def("getIntOrFloatBitWidth", nullcheck(&getIntOrFloatBitWidth))
      .def("isSignlessIntOrIndex", nullcheck(&Type::isSignlessIntOrIndex))
      .def("isSignlessIntOrIndexOrFloat",
           nullcheck(&Type::isSignlessIntOrIndexOrFloat))
      .def("isSignlessIntOrFloat", nullcheck(&Type::isSignlessIntOrFloat))
      .def("isIntOrIndex", nullcheck(&Type::isIntOrIndex))
      .def("isIntOrFloat", nullcheck(&Type::isIntOrFloat))
      .def("isIntOrIndexOrFloat", nullcheck(&Type::isIntOrIndexOrFloat));
}

} // end namespace py
} // end namespace mlir
