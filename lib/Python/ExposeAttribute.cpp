#include "Expose.h"
#include "Attribute.h"
#include "Location.h"
#include "Utility.h"
#include "Context.h"

#include <mlir/IR/Types.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/IntegerSet.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

using namespace pybind11;
using namespace llvm;
using namespace mlir;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "attribute");
}

template <typename T> auto isa() {
  return nullcheck(::isa<Attribute, T>());
}

void exposeAttribute(module &m) {
  class_<Attribute> attr{m, "Attribute"};
  attr
      .def(init<>())
      .def(self == self)
      .def(self != self)
      .def("__repr__", StringPrinter<Attribute>{})
      .def("__bool__", &Attribute::operator bool)
      .def("__invert__", &Attribute::operator!)
      .def("__hash__", overload<hash_code(Attribute)>(&hash_value))
      .def_property_readonly("kind", nullcheck(&Attribute::getKind))
      .def_property_readonly("type", nullcheck(&Attribute::getType))
      .def_property_readonly("dialect", nullcheck(&Attribute::getDialect),
                             return_value_policy::reference)
      .def("isAffineMap", isa<AffineMapAttr>())
      .def("isArray", isa<ArrayAttr>())
      .def("isBool", isa<BoolAttr>())
      .def("isDictionary", isa<DictionaryAttr>())
      .def("isFloat", isa<FloatAttr>())
      .def("isInteger", isa<IntegerAttr>())
      .def("isIntegerSet", isa<IntegerSetAttr>())
      .def("isOpaque", isa<OpaqueAttr>())
      .def("isString", isa<StringAttr>())
      .def("isSymbolRef", isa<SymbolRefAttr>())
      .def("isType", isa<TypeAttr>())
      .def("isUnit", isa<UnitAttr>())
      .def("isElements", isa<ElementsAttr>())
      .def("isDenseElements", isa<DenseElementsAttr>())
      .def("isDenseStringElements", isa<DenseStringElementsAttr>())
      .def("isDenseIntOrFPElements", isa<DenseIntOrFPElementsAttr>())
      .def("isDenseFPElements", isa<DenseFPElementsAttr>())
      .def("isDenseIntElements", isa<DenseIntElementsAttr>())
      .def("isSparseElementsAttr", isa<SparseElementsAttr>())
      .def("isSplatElements", isa<SplatElementsAttr>());
  /// Location must be registered early for default arguments.
  exposeLocation(m, attr);

  class_<AffineMapAttr>(m, "AffineMapAttr", attr)
      .def(init(&AffineMapAttr::get))
      .def("getValue", nullcheck(&AffineMapAttr::getValue));

  exposeArrayAttr(m, attr);

  class_<BoolAttr>(m, "BoolAttr", attr)
      .def(init([](bool value)
                { return BoolAttr::get(value, getMLIRContext()); }))
      .def("getValue", nullcheck(&BoolAttr::getValue));

  exposeDictAttr(m, attr);
  exposeIntFPAttr(m, attr);

  // TODO IntegerSet
  class_<IntegerSetAttr>(m, "IntegerSetAttr", attr)
      .def(init(&IntegerSetAttr::get))
      .def("getValue", nullcheck(&IntegerSetAttr::getValue));

  // TODO OpaqueAttr
  class_<OpaqueAttr>(m, "OpaqueAttr", attr)
      .def(init(&getOpaqueAttr), "dialect"_a, "data"_a, "type"_a = Type{},
                                 "location"_a = getUnknownLoc())
      .def_property_readonly("dialect", nullcheck(&getOpaqueAttrDialect))
      .def_property_readonly("data", nullcheck(&getOpaqueAttrData));

  class_<StringAttr>(m, "StringAttr", attr)
      .def(init([](const std::string &bytes, Type ty) {
        if (ty)
          return StringAttr::get(bytes, ty);
        return StringAttr::get(bytes, getMLIRContext());
      }), "bytes"_a, "type"_a = Type{})
      .def("getValue", nullcheck([](StringAttr attr) {
                                 return attr.getValue().str(); }));

  exposeSymbolRefAttr(m, attr);

  class_<TypeAttr>(m, "TypeAttr", attr)
      .def(init(&TypeAttr::get))
      .def_property_readonly("type", nullcheck(&TypeAttr::getValue));

  class_<UnitAttr>(m, "UnitAttr", attr)
      .def(init([]() { return UnitAttr::get(getMLIRContext()); }));

  exposeElementsAttr(m, attr);

  implicitly_convertible_from_all<Attribute,
      AffineMapAttr, ArrayAttr, BoolAttr, DictionaryAttr, FloatAttr,
      IntegerAttr, IntegerSetAttr, OpaqueAttr, StringAttr, SymbolRefAttr,
      FlatSymbolRefAttr, TypeAttr,

      ElementsAttr, DenseElementsAttr, DenseStringElementsAttr,
      DenseIntOrFPElementsAttr, DenseFPElementsAttr,
      DenseIntElementsAttr>(attr);
}

} // end namespace py
} // end namespace mlir
