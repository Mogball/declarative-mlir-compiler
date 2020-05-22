#include "Context.h"
#include "Location.h"
#include "Support.h"
#include "Type.h"
#include "Expose.h"

#include <mlir/IR/StandardTypes.h>

using namespace mlir;
using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "type");
}

void exposeStandardNumericTypes(pybind11::module &m, TypeClass &type) {
  class_<ComplexType>(m, "ComplexType", type)
      .def(init([](Type elTy) {
        return ComplexType::getChecked(elTy, getUnknownLoc());
      }))
      .def(init(&ComplexType::getChecked))
      .def_property_readonly("elementType",
                             nullcheck(&ComplexType::getElementType));

  class_<IndexType>(m, "IndexType", type)
      .def(init([]() { return IndexType::get(getMLIRContext()); }));

  class_<IntegerType> intType{m, "IntegerType", type};
  intType
      .def(init([](unsigned width) {
        return IntegerType::getChecked(width, getUnknownLoc());
      }))
      .def(init([](unsigned width, IntegerType::SignednessSemantics signedness) {
        return IntegerType::getChecked(width, signedness, getUnknownLoc());
      }))
      .def(init(overload<IntegerType(unsigned, Location)>(
          &IntegerType::getChecked)))
      .def(init(overload<IntegerType(unsigned, IntegerType::SignednessSemantics,
                                     Location)>(&IntegerType::getChecked)))
      .def_property_readonly("width", nullcheck(&IntegerType::getWidth))
      .def_property_readonly("signedness", nullcheck(&IntegerType::getSignedness))
      .def("isSignless", nullcheck(&IntegerType::isSignless))
      .def("isSigned", nullcheck(&IntegerType::isSigned))
      .def("isUnsigned", nullcheck(&IntegerType::isUnsigned));

  enum_<IntegerType::SignednessSemantics>(intType, "SignednessSemantics")
      .value("Signless", IntegerType::Signless)
      .value("Signed", IntegerType::Signed)
      .value("Unsigned", IntegerType::Unsigned)
      .export_values();

  class_<FloatType>(m, "FloatType", type)
      .def_property_readonly("width", nullcheck(&FloatType::getWidth));
      // llvm::fltSemantics definition not publicly visible

  m.def("BF16", []() { return FloatType::getBF16(getMLIRContext()); });
  m.def("F16", []() { return FloatType::getF16(getMLIRContext()); });
  m.def("F32", []() { return FloatType::getF32(getMLIRContext()); });
  m.def("F64", []() { return FloatType::getF64(getMLIRContext()); });

  class_<NoneType>(m, "NoneType", type)
      .def(init([]() { return NoneType::get(getMLIRContext()); }));
}

} // end namespace py
} // end namespace mlir
