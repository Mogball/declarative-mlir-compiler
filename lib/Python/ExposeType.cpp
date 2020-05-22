#include "Type.h"
#include "Expose.h"
#include "Support.h"

#include <mlir/IR/Dialect.h>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

using namespace pybind11;
using namespace llvm;
using namespace mlir;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "type");
}

void exposeType(module &m) {
  class_<Type> type{m, "Type"};
  type
      .def(init<>())
      .def(init<const Type &>())
      .def(self == self)
      .def(self != self)
      .def("__repr__", StringPrinter<Type>{})
      .def("__bool__", &Type::operator bool)
      .def("__invert__", &Type::operator!)
      .def("__hash__", overload<hash_code(Type)>(&hash_value))
      .def_property_readonly("kind", nullcheck(&Type::getKind))
      .def_property_readonly("dialect", nullcheck(&Type::getDialect),
                             return_value_policy::reference)
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

  exposeFunctionType(m, type);
  exposeOpaqueType(m, type);

  implicitly_convertible_from_all<Type,
      FunctionType, OpaqueType>(type);
}

} // end namespace py
} // end namespace mlir

namespace pybind11 {

template <> struct polymorphic_type_hook<Type>
    : public polymorphic_type_hooks<Type,
      FunctionType, OpaqueType> {};

} // end namespace pybind11
