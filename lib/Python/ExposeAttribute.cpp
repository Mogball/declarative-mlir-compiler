#include "Attribute.h"
#include "Support.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

using namespace pybind11;
using namespace llvm;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "attribute");
}

void exposeAttribute(module &m) {
  class_<Attribute>(m, "Attribute")
      .def(init<>())
      .def(init<const Attribute &>())
      .def(self == self)
      .def(self != self)
      .def("__repr__", StringPrinter<Attribute>{})
      .def("__bool__", &Attribute::operator bool)
      .def("__invert__", &Attribute::operator!)
      .def("__hash__", overload<hash_code(Attribute)>(&hash_value))
      .def_property_readonly("kind", nullcheck(&Attribute::getKind))
      .def_property_readonly("type", nullcheck(&Attribute::getType));
}

} // end namespace py
} // end namespace mlir
